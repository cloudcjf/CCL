import random 
import pickle 
import itertools 
import numpy as np 
# import matplotlib.pyplot as plt 
from torchpack.utils.config import configs
import copy 

class Memory:
    def __init__(self):
        self.K = configs.train.memory.num_pairs # 256
        self.train_tuples = []
        self.tuple_env_idx = [] # Records env tuples were stored from 

    def __len__(self):
        return len(self.train_tuples)

    def get_tuples(self, new_dataset_len = 0):
        tuples = copy.deepcopy(list(itertools.chain.from_iterable(self.train_tuples)))

    
        # Adjust id, positives, non_negatives to match dataset we'll be appending to 
        for t in tuples:
            t.id = t.id + new_dataset_len
            t.positives = t.positives + new_dataset_len
            t.non_negatives = t.non_negatives + new_dataset_len

        tuples_dict = {t.id: t for t in tuples}

        return tuples_dict

    def adjust_positive_non_negative_idx(self, env_replaced_idx):
        # Get the tuples from a new environment
        '''
            self.train_tuples: save anchor and positive tuples in a list
            env_tuples: seperately save anchor and positive tuples
            len(env_tuples) = len(self.train_tuples)
        '''
        env_tuples = list(itertools.chain.from_iterable([self.train_tuples[i] for i in env_replaced_idx]))
        # Get dict connecting id in old dataset to id in memory bank
        # old_idx: id idxs
        # new_idx: a list from 0 to 511
        old_idx = [t.id for t in env_tuples]
        new_idx = list(itertools.chain.from_iterable([2*x, 2*x + 1] for x in env_replaced_idx))
        assert(len(old_idx) == len(new_idx))
        # construct a dict to correspond ids with idxs
        old_to_new_id = {o:n for o,n in zip(old_idx, new_idx)}

        # Replace all the positives and non_negatives with new idx
        for idx, t in enumerate(env_tuples):
            positives = t.positives 
            non_negatives = t.non_negatives

            new_id = old_to_new_id[t.id]
            new_positives = [old_to_new_id[p] for p in positives if p in old_to_new_id.keys()]
            new_non_negatives = [old_to_new_id[p] for p in non_negatives if p in old_to_new_id.keys()]

            t.id = new_id
            t.positives = np.sort(new_positives)
            t.non_negatives = np.sort(new_non_negatives)

            env_tuples[idx] = t



        env_tuples_paired = [[env_tuples[x], env_tuples[x+1]] for x in list(range(len(env_tuples)))[::2]]
        assert len(env_replaced_idx) == len(env_tuples_paired)
        for pair, replace_idx in zip(env_tuples_paired, env_replaced_idx):
            self.train_tuples[replace_idx] = pair 


    def update_memory(self, new_pickle, env_idx):
        
        # Load new tuples
        new_tuples = pickle.load(open(new_pickle, 'rb'))
        new_tuples_idx = list(range(len(new_tuples)))
        random.shuffle(new_tuples_idx) # Randomly shuffle the order of new tuples for selection
        num_to_replace = self.K // (env_idx + 1)
        num_replaced = 0
        selected_idx = [] # List of already selected positives; prevent double dipping!
        env_replaced_idx = [] # List of replaced idx in the memory
        # Replace tuples 
        while(num_replaced < num_to_replace):
            # Get new tuple pair to append to list 
            anchor_idx = new_tuples_idx.pop(0)
            if anchor_idx in selected_idx: # Skip if already been selected
                continue
            anchor_tuple = new_tuples[anchor_idx]
            # positive_idxs of anchor_idx
            pair_idx_possibilities = [x for x in anchor_tuple.positives if x not in selected_idx]

            # Check a valid positive pair is possible 
            if len(pair_idx_possibilities) == 0:
                continue 

            pair_idx = random.choice(pair_idx_possibilities)
            pair_tuple = new_tuples[pair_idx]

            selected_idx += [anchor_idx, pair_idx] # Prevent these being picked again

            # Get replace idx 
            if len(self.train_tuples) < self.K: # Just fill up if less than K pairs in memory
                self.train_tuples.append([anchor_tuple, pair_tuple])
                self.tuple_env_idx.append(env_idx)
                env_replaced_idx.append(len(self.train_tuples) - 1)
            else: # Find and replace most represented environment
                x = self.tuple_env_idx[self.tuple_env_idx != env_idx]
                unique, counts = np.unique(x, return_counts = True)
                replace_env = np.argmax(counts)
                replace_idx = np.random.choice(np.nonzero(self.tuple_env_idx == replace_env)[0])
                env_replaced_idx.append(replace_idx)
                self.train_tuples[replace_idx] = [anchor_tuple, pair_tuple]
                self.tuple_env_idx[replace_idx] = env_idx 

            num_replaced += 1 
            if len(new_tuples_idx) == 0:
                print(f'Warning: Ran out of examples when adding memory for pickle file {new_pickle} at environment # {env_idx}')
                break 

        self.adjust_positive_non_negative_idx(env_replaced_idx)
