# Author: Jacek Komorowski
# Warsaw University of Technology
# Modified: Jiafeng Cui

import random 
import torch
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from torchpack.utils.config import configs 
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from datasets.oxford import OxfordDataset, TrainTransform, TrainSetTransform
from datasets.samplers_inc import BatchSampler


def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    lidar_pc = lidar_pc.numpy()
    lidar_pc = np.hstack((lidar_pc, np.zeros((len(lidar_pc),1), dtype=np.float32)))
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    # inputs = sparse_collate([inputs])
    # inputs.C = inputs.C.int()
    if return_points:
        return inputs , feats
    else:
        return inputs


def sparcify_and_collate_list(list_data, voxel_size):
    outputs = []
    for xyzr in list_data:
        outputs.append(make_sparse_tensor(xyzr, voxel_size))
    outputs =  sparse_collate(outputs)
    outputs.C = outputs.C.int()
    return outputs


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
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)
        anchor_size = int(len(labels)/4)
        anchors = batch[:anchor_size,:,:]
        positives = batch[anchor_size:2*anchor_size,:,:]
        queries = torch.cat((anchors, positives), 0)
        keys = torch.cat((positives, anchors), 0)
        memories = batch[2*anchor_size:,:,:]

        if configs.model.name == 'PointNetVlad':
            queries = {'cloud': queries}
            keys = {'cloud': keys}
            memories = {'cloud': memories}
        elif configs.model.name == 'logg3d':
            queries = sparcify_and_collate_list(queries, mink_quantization_size)
            keys = sparcify_and_collate_list(keys, mink_quantization_size)
            memories = sparcify_and_collate_list(memories, mink_quantization_size)
            queries = {'cloud': queries}
            keys = {'cloud': keys}
            memories = {'cloud': memories}
        else:
            coords_queries = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                    for e in queries]
            coords_queries = ME.utils.batched_coordinates(coords_queries)
            coords_keys = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                    for e in keys]
            coords_keys = ME.utils.batched_coordinates(coords_keys)
            coords_memories = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                    for e in memories]
            coords_memories = ME.utils.batched_coordinates(coords_memories)
            feats_queries = torch.ones((coords_queries.shape[0], 1), dtype=torch.float32)
            feats_keys = torch.ones((coords_keys.shape[0], 1), dtype=torch.float32)
            feats_memories = torch.ones((coords_memories.shape[0], 1), dtype=torch.float32)
            queries = {'coords': coords_queries, 'features': feats_queries, 'cloud': queries}
            keys = {'coords': coords_keys, 'features': feats_keys, 'cloud': keys}
            memories = {'coords': coords_memories, 'features': feats_memories, 'cloud': memories}
        # index order of queies is same as labels
        return queries, keys, memories, labels

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
    dataset.add_memory(memory)
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