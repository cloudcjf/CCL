import os 
import torch
import argparse 
import random 
import numpy as np 
import itertools 

from torchpack.utils.config import configs 

from eval.evaluate import evaluate 
from trainer_contrastive import Trainer

from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    # Repeatability 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = "config/protocols/2-step.yaml", required = False, help = 'Path to configuration YAML file')
    parser.add_argument('--train_environment', type = str, default = "pickles/Oxford/Oxford_train_queries.pickle", required = False, help = 'Path to training environment pickle')
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        # decrease the learning rate at certain epochs
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    if not os.path.exists(configs.save_dir):
        os.makedirs(configs.save_dir)
    logger = SummaryWriter(os.path.join(configs.save_dir, 'tf_logs'))

    # Train model
    trainer = Trainer(logger,args.train_environment,configs.save_dir)
    trained_model = trainer.train()

    # Evaluate 
    eval_stats = evaluate(trained_model, -1)

    # Save model
    ckpt = trained_model.state_dict()
    torch.save(ckpt, os.path.join(configs.save_dir, 'final_ckpt.pth'))
