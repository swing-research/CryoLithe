"""
Polynomial ramp model for the filters
"""

import torch
import torch.nn as nn


class PolynomialModel(nn.Module):
    def __init__(self, degree: int):
        super(PolynomialModel, self).__init__()
        self.degree = degree

        self.poly_parameters = torch.nn.Parameter(torch.rand(degree + 1))
        
    def forward(self, res: int):
        """
        Resolution of the polynomial model
        """
        x = torch.linspace(0, 1, res).to(self.poly_parameters.device)
        # Flip the x value to get a symmetric polynomial

        ramp = x*0 + self.poly_parameters[0]

        for i in range(1, self.degree + 1):
            ramp += self.poly_parameters[i] * x**i

        # Flip and concatenate the ramp
        ramp = torch.cat([ramp, ramp.flip(0)])

        ramp = ramp / ramp.max()

        return ramp