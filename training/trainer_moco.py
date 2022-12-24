import os
import random
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
    def __init__(self, logger, train_environment, save_dir):
        # Initialise inputs
        super(Trainer, self).__init__()
        self.debug = configs.debug 
        self.logger = logger 
        self.save_dir = save_dir
        self.epochs = configs.train.optimizer.epochs 
        # Set up meters and stat trackers 
        self.loss_contrastive_meter = AverageMeter()
        self.positive_score_meter = AverageMeter()
        self.negative_score_meter = AverageMeter()
        # moco
        self.K = 10000
        self.m = 0.99
        self.T = 0.07
        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = train_environment, memory = None)

        # Build model
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_q = model_factory(ckpt = None, device = 'cuda')
        self.model_k = model_factory(ckpt = None, device = 'cuda')
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue_pcd', torch.randn(128, len(self.dataloader.dataset.queries)))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0).cuda()
        self.queue_pcd_index = set(range(len(self.dataloader.dataset.queries)))

        # make cross entropy loss
        self.criterion = nn.CrossEntropyLoss().cuda()

        # Make optimizer 
        if configs.train.optimizer.name == "SGD":
            self.optimizer = torch.optim.SGD(self.model_q.parameters(), lr=configs.train.optimizer.lr,
                                        momentum=configs.train.optimizer.momentum,
                                        weight_decay=configs.train.optimizer.weight_decay)
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
    def _dequeue_and_enqueue_pcd_fast(self, keys, labels):
        self.queue_pcd[:, labels] = keys.T


    '''
        reset epoch metrics
    '''
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_contrastive_meter.reset()
        self.positive_score_meter.reset()
        self.negative_score_meter.reset()

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
        # Prepare batch
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
        queue_pcd_clone = self.queue_pcd.clone().detach()
        negatives_list = []
        for label in labels:
            negative_index = random.sample(list(self.queue_pcd_index.difference(set(self.dataloader.dataset.queries[label].non_negatives))), self.K)
            negatives_list.append(queue_pcd_clone[:,negative_index])
        negatives_tensor = torch.stack(negatives_list, dim=0)
        l_neg_pcd = torch.einsum('nc,nck->nk', [projectors, negatives_tensor])
        positive_score = torch.mean(l_pos_pcd)
        negative_score = torch.mean(l_neg_pcd)
        logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd],dim=1)
        logits_pcd /= self.T
        labels_pcd = torch.zeros(logits_pcd.shape[0], dtype=torch.long).cuda()
        loss = self.criterion(logits_pcd, labels_pcd)
        # dequeue and enqueue
        self._dequeue_and_enqueue_pcd_fast(key_projectors, key_labels)

        # Backwards
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_contrastive_meter.update(loss.item())
        self.positive_score_meter.update(positive_score.item())
        self.negative_score_meter.update(negative_score.item())
        return None 


    # 1. update learning rate
    # 2. update batch size
    # 3. save log data
    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()
        # for tag, value in self.model_q.named_parameters():
        #     tag = tag.replace('.', '/')
        #     self.logger.add_histogram(tag, value.data.cpu().numpy(), epoch)
        #     self.logger.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

        # Tensorboard plotting 
        self.logger.add_scalar(f'Contrastive_Loss_epoch', self.loss_contrastive_meter.avg, epoch)
        self.logger.add_scalar(f'positive_score_epoch', self.positive_score_meter.avg, epoch)
        self.logger.add_scalar(f'negative_score_epoch', self.negative_score_meter.avg, epoch)

    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (queries, keys, labels) in enumerate(self.dataloader):
                self.training_step(queries, keys, labels)
                if self.debug and idx > 2:
                    break
            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break

        return self.model_q
