import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
import numpy as np 
from tqdm import tqdm 
from eval.evaluate import evaluate 
from misc.util import AverageMeter
from models.model_factory import model_factory
from datasets.dataset_util import make_dataloader
from torchpack.utils.config import configs 


class Trainer(nn.Module):
    def __init__(self, logger, train_environment):
        # Initialise inputs
        super(Trainer, self).__init__()
        self.debug = configs.debug 
        self.logger = logger 
        self.epochs = configs.train.optimizer.epochs 
        # Set up meters and stat trackers 
        self.loss_contrastive_meter = AverageMeter()
        self.alpha = 0.3
        self.beta = 0.5
        # moco
        # self.K = 15000
        self.K = 5000
        self.m = 0.99
        self.T = 0.07

        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = train_environment, memory = None)

        # Build model
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        # pretrained_checkpoint = torch.load("/home/ps/cjf/InCloud/weights/minkloc3d_baseline.pth")
        self.model_q = model_factory(ckpt = None, device = 'cuda')
        self.model_k = model_factory(ckpt = None, device = 'cuda')
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue_pcd', torch.randn(128, self.K))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0).cuda()
        self.register_buffer("queue_pcd_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_pcd_index = [-1]*self.K
        # Make loss functions
        self.criterion = nn.CrossEntropyLoss().cuda()

        # Make optimizer 
        if configs.train.optimizer.weight_decay is None or configs.train.optimizer.weight_decay == 0:
            self.optimizer = torch.optim.Adam(self.meters(), lr=configs.train.optimizer.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model_q.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

        # Scheduler
        if configs.train.optimizer.scheduler is None:
            self.scheduler = None
        else:
            if configs.train.optimizer.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=configs.train.optimizer.epochs+1,
                                                                    eta_min=configs.train.optimizer.min_lr)
            elif configs.train.optimizer.scheduler == 'MultiStepLR':
                if not isinstance(configs.train.optimizer.scheduler_milestones, list):
                    configs.train.optimizer.scheduler_milestones = [configs.train.optimizer.scheduler_milestones]
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, configs.train.optimizer.scheduler_milestones, gamma=0.1)
            else:
                raise NotImplementedError('Unsupported LR scheduler: {}'.format(configs.train.optimizer.scheduler))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue_pcd(self, keys, labels):
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            keys = concat_all_gather(keys)
            labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_pcd_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue_pcd[:, ptr:ptr + batch_size] = keys.T
            self.queue_pcd_index[ptr:ptr + batch_size] = labels
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue_pcd[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue_pcd[:, :head_size] = keys.T[:, tail_size:]
            self.queue_pcd_index[ptr:self.K] = labels[:tail_size]
            self.queue_pcd_index[:head_size] = labels[tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_pcd_ptr[0] = ptr


    '''
        reset epoch metrics
    '''
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_contrastive_meter.reset()

    '''
        1. clear the gradient of weights -------> self.optimizer.zero_grad()
        2. model forward
        3. calculate the loss
        4. calculate the gradients -------------> loss.backward()
        5. update the weights ------------------> self.optimizer.step()
        6. empty CUDA cache
        7. update epoch metrics
    '''
    def training_step(self, queries, keys, labels):
        queries = {x: queries[x].to('cuda') if x!= 'coords' else queries[x] for x in queries}
        keys = {x: keys[x].to('cuda') if x!= 'coords' else keys[x] for x in keys}
        half_size = int(len(labels)/2)
        key_labels = []
        key_labels += labels[half_size:]
        key_labels += labels[:half_size]
        # Get embeddings and Loss
        self.optimizer.zero_grad()
        embeddings, projectors = self.model_q(queries)
        projectors = nn.functional.normalize(projectors, dim=1)
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            _, key_projectors = self.model_k(keys)
            key_projectors = nn.functional.normalize(key_projectors, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_pcd = torch.einsum('nc,nc->n', [projectors, key_projectors]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_pcd = torch.einsum('nc,ck->nk', [projectors, self.queue_pcd.clone().detach()])

        # choose the true negatives
        for i in range(len(labels)):
            false_negative_index = [e for e in range(len(self.queue_pcd_index)) if self.queue_pcd_index[e] in self.dataloader.dataset.queries[labels[i]].non_negatives]
            if len(false_negative_index) > 0:
                l_neg_pcd[i,false_negative_index] = 0

        # cjf faster version
        logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd],dim=1)
        logits_pcd /= self.T
        labels_pcd = torch.zeros(logits_pcd.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(logits_pcd, labels_pcd)
        # cjf faster version

        # new loss
        # logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd],dim=1)
        # most_similar = torch.max(logits_pcd)
        # positive_loss = 1 - l_pos_pcd
        # positive_loss = torch.mean(positive_loss)
        # l_neg_pcd = torch.where(l_neg_pcd > self.beta, l_neg_pcd, torch.tensor(0.0).cuda())
        # num_negative = l_neg_pcd.norm(0)
        # if num_negative > 0:
        #     negative_loss = torch.sum(l_neg_pcd) / num_negative
        # else:
        #     negative_loss = 0
        # regular_loss = -torch.log((1-most_similar)/2)
        # loss = positive_loss + negative_loss
        # new loss

        # dequeue and enqueue
        self._dequeue_and_enqueue_pcd(key_projectors, key_labels)

        # Backwards
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_contrastive_meter.update(loss.item())
        return None 


    # 1. update learning rate
    # 2. save log data
    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Tensorboard plotting 
        self.logger.add_scalar(f'Contrastive_Loss_epoch', self.loss_contrastive_meter.avg, epoch)

    # for loop on epochs
    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (queries, keys, labels) in enumerate(self.dataloader):
                self.training_step(queries, keys, labels)

            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break 

        return self.model_q
