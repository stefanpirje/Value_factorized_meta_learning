import os
import random
import numpy as np
import torch
from torch import nn

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def sampler(B, trajectories, max_batch_num):
    permutation = torch.randperm(trajectories[0].size(0))
    batch_idxs = torch.split(permutation, B)  # list of batches of transition idxs
    batch_idxs = batch_idxs[: (max_batch_num or len(batch_idxs))]
    for idxs in batch_idxs:
        yield [el[idxs, :] for el in trajectories]

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
      if x.dim() == 2:
          bias = self._bias.t().view(1, -1)
      else:
          bias = self._bias.t().view(1, -1, 1, 1)
      return x + bias
    
def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr