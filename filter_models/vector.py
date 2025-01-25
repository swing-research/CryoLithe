"""
This script contains the vector model for the filter.
"""

import torch
import torch.nn as nn
from skimage.transform.radon_transform import _get_fourier_filter

class VectorModel(nn.Module):
    def __init__(self, init: str, size: int, symmetric: bool = True):
        super(VectorModel, self).__init__()
        self.size = size
        self.init = init
        self.symmetric = symmetric

        if self.init == 'ones':
            self.vector = nn.Parameter(torch.ones(size))
        else:
            vector = _get_fourier_filter(2*size, init)[:size,0]
            self.vector = nn.Parameter(torch.tensor(vector, dtype=torch.float32))

    def forward(self, x: int):
        if self.symmetric:
            return torch.cat([self.vector,self.vector.flipud()])
        return self.vector


# class VectorModel_symmetric(nn.Module):
#     """
#     Corrected symmetric version
#     """
#     def __init__(self, init: str, size: int):
#         super(VectorModel_symmetric, self).__init__()
#         self.size = size
#         self.init = init

#         if self.init == 'ones':
#             self.vector = nn.Parameter(torch.ones(size+1))
#         else:
#             vector = _get_fourier_filter(2*size, init)[:size+1,0]
#             self.vector = nn.Parameter(torch.tensor(vector, dtype=torch.float32))

#     def forward(self, x: int):
#         return torch.cat([self.vector,self.vector.flipud()[1:-1]])


class VectorModel_symmetric(nn.Module):
    """
    Corrected symmetric version
    """
    def __init__(self, init: str, size: int, linear_filter: bool = False):
        super(VectorModel_symmetric, self).__init__()
        self.size = size
        self.init = init
        self.linear_filter = linear_filter

        if self.init == 'ones':
            self.vector = nn.Parameter(torch.ones(size+1))
        else:
            vector = _get_fourier_filter(2*size, init)[:size+1,0]
            self.vector = nn.Parameter(torch.tensor(vector, dtype=torch.float32))

    def forward(self, x: int):
        if self.linear_filter:
            filter_value = torch.cat([self.vector,self.vector.flipud()[1:-1]])
            filter_value = torch.fft.fftshift(torch.fft.ifft(filter_value))
            filter_value[:self.size//2] = 0
            filter_value[-self.size//2:] = 0
            filter_value = torch.fft.fft(torch.fft.ifftshift(filter_value))
            return filter_value
        else:
            return torch.cat([self.vector,self.vector.flipud()[1:-1]])



class VectorModel_real(nn.Module):
    """
    Corrected symmetric version
    """
    def __init__(self, init: str, size: int):
        super(VectorModel_real, self).__init__()
        self.size = size
        self.init = init

        if self.init == 'ones':
            self.vector = nn.Parameter(torch.ones(size+1))
        else:
            vector = torch.tensor(_get_fourier_filter(size, init)[:,0], dtype=torch.float32)
            vector = torch.fft.ifft(vector).real
            self.vector = nn.Parameter(vector)

    def forward(self, x: int):
        """
        x: size of the projection along one dimension
        Note that output should be twice the size of x, as we we use double the size of the FFT
        """

        response_real = torch.zeros(2*x,dtype=torch.float32).to(self.vector.device)
        response_size = min(self.size,x)
        response_real[:response_size//2] = self.vector[:response_size//2]
        response_real[-response_size//2:] = self.vector[-response_size//2:]
        return torch.fft.fft(response_real)

