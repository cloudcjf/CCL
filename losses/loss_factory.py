from torchpack.utils.config import configs 
# from losses.pr_loss import *
from losses.inc_loss import * 
from losses.contrastive_loss import * 

def make_pr_loss():
    pr_loss_name = configs.train.loss.pr.name 
    if pr_loss_name == 'BatchHardTripletMarginLoss':
        # BatchHard mining with triplet margin loss
        # Expects input: embeddings, positives_mask, negatives_mask
        loss_fn = BatchHardTripletLossWithMasks(configs.train.loss.pr.margin, configs.model.normalize_embeddings)
    elif pr_loss_name == 'BatchHardContrastiveLoss':
        loss_fn = BatchHardContrastiveLossWithMasks(configs.train.loss.pr.pos_margin, configs.train.loss.pr.neg_margin, configs.model.normalize_embeddings)
    else:
        print('Unknown loss: {}'.format(pr_loss_name))
        raise NotImplementedError
    return loss_fn

def make_inc_loss():
    inc_loss_name = configs.train.loss.incremental.name 
    if inc_loss_name == None or inc_loss_name == 'None':
        loss_fn = NoIncLoss()
    elif inc_loss_name == 'LwF':
        loss_fn = LwF()
    elif inc_loss_name == 'EWC':
        loss_fn = EWC() 
    elif inc_loss_name == 'StructureAware':
        loss_fn = StructureAware()
    elif inc_loss_name == 'KD':
        loss_fn = KD()
    else:
        raise NotImplementedError(f'Unknown Loss : {inc_loss_name}')
    return loss_fn

def make_contrastive_loss():
    # 1018 0.03 is the best
    loss_fn = ContrastiveLoss(temperature=0.07)
    return loss_fn