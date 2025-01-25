"""
This script contains the FIR filter to be applied on the projections
"""

import torch
import torch.nn as nn
from skimage.transform.radon_transform import _get_fourier_filter

class FIRModel(nn.Module):
    def __init__(self, init: str, size: int):
        super(FIRModel, self).__init__()
        self.size = size
        self.init = init

        if self.init == 'ones':
            self.fir  = nn.Parameter(torch.ones(size,size))
        if self.init == 'impulse':
            fir = torch.zeros(size,size,dtype=torch.float32)
            fir[size//2,size//2] = 1
            self.fir =  nn.Parameter(fir)
        else:
            self.fir  =  nn.Parameter(torch.randn(size,size))

    def forward(self, x: int):
        return self.fir
