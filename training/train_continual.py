import os 
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import torch 
import argparse 
import random 
import numpy as np 
import itertools 

from torchpack.utils.config import configs 
from datasets.memory import Memory

from eval.evaluate import evaluate 
from eval.metrics import IncrementalTracker 
from trainer_continual import TrainerIncremental

from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':

    # Repeatability 
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Get args and configs 
    parser = argparse.ArgumentsParser()
    parser.add_argument('--initial_ckpt', type = str, default = "/heyufei1/models/ccl/cjf_results/oxford_mink/final_ckpt.pth", required = False, help = 'Path to first environment in incremental sequence.')
    parser.add_argument('--initial_environment', type = str, default = "pickles/Oxford/Oxford_train_queries.pickle", required = False, help = 'Path to training pickle for the first environment')
    parser.add_argument('--incremental_environments', type = str, default = "pickles/In-house/In-house_train_queries.pickle", required = True, nargs = '+', help = 'train on inhouse')
    parser.add_argument('--config', type = str, default = "config/protocols/4-step.yaml", required = False, help = 'Path to configuration YAML file')

    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive = True)
    configs.update(opts)
    if isinstance(configs.train.optimizer.scheduler_milestones, str):
        # decrease the learning rate at certain epochs
        configs.train.optimizer.scheduler_milestones = [int(x) for x in configs.train.optimizer.scheduler_milestones.split(',')] # Allow for passing multiple drop epochs to scheduler
    print(configs)

    # Make save directory and logger
    save_dir = "/heyufei1/models/ccl/cjf_results/mink_projector_4steps"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, 'models')):
        os.makedirs(os.path.join(save_dir, 'models'))
    logger = SummaryWriter(os.path.join(save_dir, 'tf_logs'))

    # Load and save initial checkpoint
    print('Loading Initial Checkpoint: ', end = '')
    assert os.path.exists(args.initial_ckpt), f'Initial Checkpoint at {args.initial_ckpt} should not be none'
    # the trained model in the last step
    old_ckpt = torch.load(args.initial_ckpt)
    torch.save(old_ckpt, os.path.join(save_dir, 'models', f'env_0.pth'))
    print('Done')

    print('Loading Memory: ', end = '')
    # Make metric tracker, incremental memory
    metrics = IncrementalTracker()
    # memory = None
    memory = Memory()

    # Update memory, metric tracker with env. 0
    '''
        args.initial_environment: pickles/Oxford/Oxford_train_queries.pickle
        env_idx: 0
        args.incremental_environments: pickles/combined_train_queries.pickle or other incremental_pickle
    '''
    memory.update_memory(args.initial_environment, env_idx = 0)
    metrics.update(configs.eval.initial_environment_result, env_idx = 0)
    print('Done')

    # Iterate over training steps
    old_env = args.initial_ckpt
    for env_idx, env in enumerate(args.incremental_environments):
        print(f'Training on environment # {env_idx + 1}')
        env_idx = env_idx + 1 # Start env_idx at 1

        # Make Trainer
        trainer = TrainerIncremental(logger, memory, old_env, env, old_ckpt, env_idx)
        torch.cuda.empty_cache() 
        old_env = env # For EWC 

        # Train on this environment
        new_model = trainer.train()
        if not configs.debug or env_idx == len(args.incremental_environments):
            eval_stats = evaluate(new_model, env_idx)
            # Update Inc. Learning Stats
            metrics.update(eval_stats, env_idx)

        # Save model
        old_ckpt = new_model.state_dict()
        torch.save(old_ckpt, os.path.join(save_dir, 'models', f'env_{env_idx + 1}.pth'))

        # Update Memory 
        memory.update_memory(env, env_idx)

    # Print and save final results
    torch.save(old_ckpt, os.path.join(save_dir, 'models', 'final_ckpt.pth'))
    results_final = metrics.get_results()
    results_final.to_csv(os.path.join(save_dir, 'results.csv'))
