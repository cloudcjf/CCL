import os 
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch 
import argparse 
import random 
import numpy as np 
import itertools 

from torchpack.utils.config import configs 

from eval.evaluate import evaluate 
from eval.metrics import IncrementalTracker 
from trainer import Trainer

from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    # Repeatability 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = "config/protocols/2-step.yaml", required = False, help = 'Path to configuration YAML file')
    # parser.add_argument('--train_environment', type = str, default = "pickles/Oxford/Oxford_train_queries.pickle", required = False, help = 'Path to training environment pickle')
    parser.add_argument('--train_environment', type = str, default = "pickles/In-house/In-house_train_queries.pickle", required = False, help = 'Path to training environment pickle')
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        # decrease the learning rate at certain epochs
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    save_dir = "/home/ps/cjf/InCloud/cjf_results/false_negative"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = SummaryWriter(os.path.join(save_dir, 'tf_logs'))

    # Train model
    trainer = Trainer(logger,args.train_environment)
    trained_model = trainer.train()

    # Evaluate 
    eval_stats = evaluate(trained_model, -1)
    # TODO Improve printing on eval part of train loop

    # Save model
    ckpt = trained_model.state_dict()
    torch.save(ckpt, os.path.join(save_dir, 'final_ckpt.pth'))
