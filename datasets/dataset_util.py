# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset, TrainTransform, TrainSetTransform
from datasets.samplers import BatchSampler
import random 

from torchpack.utils.config import configs 



def make_dataset(pickle_file):
    # Create training and validation datasets

    datasets = {}
    train_transform = TrainTransform(configs.data.aug_mode)
    train_set_transform = TrainSetTransform(configs.data.aug_mode)

    print(f'Creating Dataset from pickle file : {pickle_file}')
    dataset = OxfordDataset(configs.data.dataset_folder, pickle_file, train_transform,
                                      set_transform=train_set_transform)

    return dataset


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        """
            data_list: the data structure follows the return of getitem
            deal with the data in batch
            it will return the data to ——> enumerate(self.dataloader)
        """
        # print("call collate function")
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)
        batch_size = batch.shape[0]
        query_size = int(batch_size/2)
        anchors = batch[:query_size,:,:]
        positives = batch[query_size:,:,:]
        queries = torch.cat((anchors, positives), 0)
        keys = torch.cat((positives, anchors), 0)


        coords_queries = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                for e in queries]
        coords_queries = ME.utils.batched_coordinates(coords_queries)
        coords_keys = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                for e in keys]
        coords_keys = ME.utils.batched_coordinates(coords_keys)
        # Assign a dummy feature equal to 1 to each point
        # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
        feats_queries = torch.ones((coords_queries.shape[0], 1), dtype=torch.float32)
        feats_keys = torch.ones((coords_keys.shape[0], 1), dtype=torch.float32)
        queries = {'coords': coords_queries, 'features': feats_queries, 'cloud': queries}
        keys = {'coords': coords_keys, 'features': feats_keys, 'cloud': keys}
        # index order of queies is same as labels
        return queries, keys, labels

    return collate_fn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(pickle_file, memory):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements

    :return:
    """

    dataset = make_dataset(pickle_file)

    train_sampler = BatchSampler(dataset, batch_size=configs.train.batch_size,
                                 batch_size_limit=configs.train.batch_size_limit,
                                 batch_expansion_rate=configs.train.batch_expansion_rate)

    # Reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(dataset, configs.model.mink_quantization_size)
    dataloader = DataLoader(dataset, batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=configs.train.num_workers, pin_memory=configs.data.pin_memory,
                                     worker_init_fn = seed_worker, generator = g)

    return dataloader
