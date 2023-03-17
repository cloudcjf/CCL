# Author: Jiafeng Cui
# cjfacl@gmail.com

import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
from tqdm import tqdm 
from eval.evaluate import evaluate 
from losses.loss_factory import make_inc_loss
from misc.util import AverageMeter
from models.model_factory import model_factory
from datasets.dataset_util_inc import make_dataloader
from torchpack.utils.config import configs


class TrainerIncremental(nn.Module):
    def __init__(self, logger, memory, old_environment_pickle, new_environment_pickle, pretrained_checkpoint, env_idx):
        # Initialise inputs
        super(TrainerIncremental, self).__init__()
        self.debug = configs.debug 
        self.logger = logger 
        self.env_idx = env_idx
        self.epochs = configs.train.optimizer.epochs 

        # Set up meters and stat trackers 
        self.loss_total_meter = AverageMeter()
        self.loss_contrastive_meter = AverageMeter()
        self.loss_inc_meter = AverageMeter()
        self.positive_score_meter = AverageMeter()
        self.negative_score_meter = AverageMeter()
        # contrastive
        self.K = 1000
        self.m = 0.99
        self.T = 0.07
        # Make dataloader
        self.dataloader = make_dataloader(pickle_file = new_environment_pickle, memory = memory)

        # Build models and init from pretrained_checkpoint
        assert torch.cuda.is_available, 'CUDA not available.  Make sure CUDA is enabled and available for PyTorch'
        self.model_frozen = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new_q = model_factory(ckpt = pretrained_checkpoint, device = 'cuda')
        self.model_new_k = model_factory(ckpt = None, device = 'cuda')
        for param_q, param_k in zip(self.model_new_q.parameters(), self.model_new_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue_pcd', torch.randn(128, len(self.dataloader.dataset.queries)))
        self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0).cuda()
        self.queue_pcd_index = set(range(len(self.dataloader.dataset.queries)))
        print("loading all embeddings")
        # loading queue
        for idx, (queries, keys, memories, labels) in enumerate(self.dataloader):
            query_size = int(len(labels)/2)
            queries = {x: queries[x].to('cuda') if x!= 'coords' else queries[x] for x in queries}
            with torch.no_grad():  # no gradient to keys
                embeddings, projectors = self.model_new_k(queries)
                projectors = nn.functional.normalize(projectors, dim=1)
            self._dequeue_and_enqueue_pcd_fast(projectors, labels[:query_size])
        print(f"finish constructing incremental_environments,size: {len(self.dataloader.dataset.queries)}")
        # Make optimizer 
        if configs.train.optimizer.name == "SGD":
            self.optimizer = torch.optim.SGD(self.model_new_q.parameters(), lr=configs.train.optimizer.lr,
                                        momentum=configs.train.optimizer.momentum,
                                        weight_decay=configs.train.optimizer.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model_new_q.parameters(), lr=configs.train.optimizer.lr, weight_decay=configs.train.optimizer.weight_decay)

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

        # Make loss functions
        self.inc_fn = make_inc_loss()
        self.criterion = nn.CrossEntropyLoss().cuda()


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_new_q.parameters(), self.model_new_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pcd_fast(self, keys, labels):
        self.queue_pcd[:, labels] = keys.T

    '''
        reset epoch metrics
    '''
    def before_epoch(self, epoch):
        # Reset meters
        self.loss_total_meter.reset()
        self.loss_inc_meter.reset()
        self.loss_contrastive_meter.reset()
        self.positive_score_meter.reset()
        self.negative_score_meter.reset()
        # Adjust weight of incremental loss function if constraint relaxation enabled
        self.inc_fn.adjust_weight(epoch)

    '''
        1. clear the gradient of weights -------> self.optimizer.zero_grad()
        2. model forward
        3. calculate the loss
        4. calculate the gradients -------------> loss.backward()
        5. update the weights ------------------> self.optimizer.step()
        6. empty CUDA cache
        7. update epoch metrics
    # '''
    def training_step(self, queries, keys, memories, labels):
        memories = {x: memories[x].to('cuda') if x!= 'coords' else memories[x] for x in memories}
        queries = {x: queries[x].to('cuda') if x!= 'coords' else queries[x] for x in queries}
        keys = {x: keys[x].to('cuda') if x!= 'coords' else keys[x] for x in keys}
        anchor_size = int(len(labels)/4)
        key_labels = labels[anchor_size:2*anchor_size] + labels[:anchor_size]
        # Get embeddings and Loss
        self.optimizer.zero_grad()
        embeddings, projectors = self.model_new_q(queries)
        projectors = nn.functional.normalize(projectors, dim=1)
        embeddings_memories, projectors_memories = self.model_new_q(memories)
        projectors_memories = nn.functional.normalize(projectors_memories, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            _, key_projectors_new = self.model_new_k(keys)
            key_projectors_new = nn.functional.normalize(key_projectors_new, dim=1)
            embeddings_memories_frozen, projectors_memories_frozen = self.model_frozen(memories)
            projectors_memories_frozen = nn.functional.normalize(projectors_memories_frozen, dim=1)

        # compute logits
        # positive logits: Nx1
        l_pos_pcd = torch.einsum('nc,nc->n', [projectors, key_projectors_new]).unsqueeze(-1)
        queue_pcd_clone = self.queue_pcd.clone().detach()
        negatives_list = []
        for index in range(len(labels[:2*anchor_size])):
            negative_index = random.sample(list(self.queue_pcd_index.difference(set(self.dataloader.dataset.queries[labels[index]].non_negatives))), self.K)
            negatives_list.append(queue_pcd_clone[:,negative_index])
        negatives_tensor = torch.stack(negatives_list, dim=0)
        # negative logits: NxK
        l_neg_pcd = torch.einsum('nc,nck->nk', [projectors, negatives_tensor])
        positive_score = torch.mean(l_pos_pcd)
        negative_score = torch.mean(l_neg_pcd)

        logits_pcd = torch.cat([l_pos_pcd, l_neg_pcd], dim=1)
        logits_pcd_contrastive = logits_pcd / self.T
        labels_pcd = torch.zeros(logits_pcd_contrastive.shape[0], dtype=torch.long).cuda()
        loss_contrastive = self.criterion(logits_pcd_contrastive, labels_pcd)

        loss_distill = self.inc_fn(projectors_memories_frozen, projectors_memories)
        loss_total = loss_contrastive + loss_distill


        # dequeue and enqueue
        self._dequeue_and_enqueue_pcd_fast(key_projectors_new, key_labels)
        # Backwards
        loss_total.backward()
        self.optimizer.step()
        torch.cuda.empty_cache() # Prevent excessive GPU memory consumption by SparseTensors

        # Stat tracking
        self.loss_total_meter.update(loss_total.item())
        self.loss_contrastive_meter.update(loss_contrastive.item())
        self.loss_inc_meter.update(loss_distill.item())
        self.positive_score_meter.update(positive_score.item())
        self.negative_score_meter.update(negative_score.item())


    '''
        # 1. update learning rate
        # 2. update batch size
        # 3. save log data
    '''
    def after_epoch(self, epoch):
        # Scheduler 
        if self.scheduler is not None:
            self.scheduler.step()

        # Tensorboard plotting
        self.logger.add_scalar(f'Step_{self.env_idx}/Total_Loss_epoch', self.loss_total_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Contrastive_Loss_epoch', self.loss_contrastive_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/Increment_Loss_epoch', self.loss_inc_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/positive_score_epoch', self.positive_score_meter.avg, epoch)
        self.logger.add_scalar(f'Step_{self.env_idx}/negative_score_epoch', self.negative_score_meter.avg, epoch)



    # for loop on epochs
    def train(self):
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.before_epoch(epoch)
            for idx, (queries, keys, memories, labels) in enumerate(self.dataloader):
                self.training_step(queries, keys, memories, labels)
                if self.debug and idx > 2:
                    break

            self.after_epoch(epoch)
            if self.debug and epoch > 2:
                break 

        return self.model_new_q
