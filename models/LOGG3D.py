# Modified: Jiafeng Cui
# remove SOP module, since it doesn't work on Oxford and In-house datasets

import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from models.netvlad import *
from models.spvnas.model_zoo import spvcnn

__all__ = ['LOGG3D']


class LOGG3D(nn.Module):
    def __init__(self, output_dim=256):
        super(LOGG3D, self).__init__()

        self.spvcnn = spvcnn(output_dim=output_dim)
        self.mp1 = torch.nn.MaxPool2d((4096, 1), 1)
        self.proj = nn.Sequential(
            nn.Linear(output_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128, bias=True)
        )

    def forward(self, batch):
        assert 'cloud' in batch.keys(), 'Error: Key "Cloud" not in batch keys.  Set model.mink_quantization_size to "None" to avoid!'
        x = batch['cloud']
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.spvcnn(x)
        y = torch.split(x, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        batch_size = x.shape[0]
        num_points = x.shape[1]
        feature_size = x.shape[2]
        #[batch, num_points, feature_dim]
        paddings = torch.zeros(batch_size,4096-num_points,feature_size).cuda()
        x = torch.cat((x,paddings),dim=1)
        x= self.mp1(x).squeeze()
        projector = self.proj(x)
        return x, projector


if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir)
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)
    lidar_pc = lidar_pc.reshape(-1, 4)
    input = make_sparse_tensor(lidar_pc, 0.05).cuda()

    model = LOGG3D().cuda()
    model.train()
    output = model(input)
    print('output size: ', output[0].size())
