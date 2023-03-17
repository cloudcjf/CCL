# Author: Jiafeng Cui
# cjfacl@gmail.com

from torchpack.utils.config import configs 
from losses.inc_loss import * 
from losses.contrastive_loss import * 


def make_inc_loss():
    inc_loss_name = configs.train.loss.incremental.name 
    if inc_loss_name == None or inc_loss_name == 'None':
        loss_fn = NoIncLoss()
    elif inc_loss_name == 'KD':
        loss_fn = KD()
    else:
        raise NotImplementedError(f'Unknown Loss : {inc_loss_name}')
    return loss_fn

def make_contrastive_loss():
    # 1018 0.03 is the best
    loss_fn = ContrastiveLoss(temperature=0.07)
    return loss_fn