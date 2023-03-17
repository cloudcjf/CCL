# Author: Jiafeng Cui
# cjfacl@gmail.com

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchpack.utils.config import configs 
from tqdm import tqdm 

class NoIncLoss:
    def __init__(self):
        pass

    def adjust_weight(self, epoch):
        pass
    
    def __call__(self, *args, **kwargs):
        return torch.tensor(0, dtype = float, device = 'cuda')

class KD:
    def __init__(self):
        self.orig_weight = configs.train.loss.incremental.weight    # λ_init
        self.weight = configs.train.loss.incremental.weight 
        self.margin = configs.train.loss.incremental.margin 
        self.t = 0.1

        # Constraint relaxation
        gamma = configs.train.loss.incremental.gamma
        lin = torch.linspace(0, 1, configs.train.optimizer.epochs)
        exponential_factor = gamma*(lin - 0.5)
        self.weight_factors = 1 / (1 + exponential_factor.exp())    # w(γ)


    def adjust_weight(self, epoch):
        if configs.train.loss.incremental.adjust_weight:
            self.weight = self.orig_weight * self.weight_factors[epoch - 1]
        else:
            pass 
    
    def __call__(self, projectors_memories_frozen, projectors_memories):
        with torch.no_grad():
            previous_similarity = torch.einsum('nc,cm->nm', [projectors_memories_frozen, projectors_memories_frozen.T])
            logits_mask = torch.scatter(torch.ones_like(previous_similarity),1,torch.arange(previous_similarity.size(0)).view(-1, 1).cuda(),0)
            q = torch.softmax(previous_similarity*logits_mask / self.t, dim=1)
        current_similarity = torch.einsum('nc,cm->nm', [projectors_memories, projectors_memories.T])
        log_p = torch.log_softmax(current_similarity*logits_mask / self.t, dim=1)
        loss_distill = self.weight * torch.nn.functional.kl_div(log_p, q, reduction="batchmean")
        return loss_distill