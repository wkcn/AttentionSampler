import mobula
import torch
import torch.nn.functional as F
from . import mobula_hack_for_pytorch


# Hack MobulaOP for the compatible functions
torch._mobula_hack = mobula_hack_for_pytorch


class AttSampler(torch.nn.Module):
    def __init__(self, scale=1.0, dense=4, iters=5):
        super(AttSampler, self).__init__()
        self.scale = scale
        self.dense = dense
        self.iters = iters

    def forward(self, data, attx, atty):
        grid = mobula.op.AttSamplerGrid(data, attx, atty,
                                        scale=self.scale,
                                        dense=self.dense,
                                        iters=self.iters)
        grid = torch.stack(grid, dim=3)
        return F.grid_sample(data, grid)
