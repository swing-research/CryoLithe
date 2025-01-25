"""
This script contains a model class which just average the center pixel of the projection patches
Used to model the adjoint operator.
"""
import torch.nn as nn
import torch
from utils.utils import generate_projections_location

class adjoint(nn.Module):
    """
    Using to obtain pixel wise estimate of the adjoint operator
    """
    def __init__(self,avg_type = 'mean'):
        super(adjoint, self).__init__()
        self.avg_type = avg_type
        

    def forward(self, x):
        """
        Forward pass of the network
        """
        patch_size = x.shape[-1]

        mid_pix = x[:,:,patch_size//2,patch_size//2]
        if self.avg_type == 'mean':
            mid_pix_sum = torch.mean(mid_pix,1)
        elif self.avg_type == 'sum':
            mid_pix_sum = torch.sum(mid_pix,1)
        else:
            raise ValueError('Invalid avg_type')
        return mid_pix_sum.unsqueeze(1)



class adjoint_patch(nn.Module):
    """
    MLP along the slices of the projection and an mlp to combine the slices with a different architecture for
    encoding and combining
    """
    def __init__(self,output_patch_size = None):
        super(adjoint_patch, self).__init__()
        self.output_patch_size = output_patch_size
        if self.output_patch_size is not None:
            assert self.output_patch_size%2 == 1, "patch size should be odd"
        

    def forward(self, x,angles):
        """
        Forward pass of the network
        """
        patch_size = x.shape[-1]
        batch_size = x.shape[0]
        device = x.device 

        x_patch = torch.linspace(-1,1,patch_size,device=device)
        y_patch = torch.linspace(-1,1,patch_size,device=device)
        z_patch = torch.linspace(-1,1,patch_size,device=device)
        xx_pathc, yy_pathc,zz_pathc = torch.meshgrid(y_patch, x_patch, z_patch, indexing='ij')
        points_patch = torch.cat([zz_pathc.unsqueeze(-1),yy_pathc.unsqueeze(-1),xx_pathc.unsqueeze(-1)],dim= -1)
        points_patch = points_patch.reshape(-1,3)
        #print(angles[0])
        points_patch_proj = generate_projections_location(points_patch,angles[0])  
        x_fbp = torch.zeros(batch_size,patch_size,patch_size,patch_size,device =device)
        # converting batch to channels
        #print(x.shape)
        x = x.permute(1,0,2,3)
        
        for i,x_proj in enumerate(x):
            x_fbp = x_fbp + torch.nn.functional.grid_sample(x_proj.unsqueeze(0),
                                points_patch_proj[i].unsqueeze(0).unsqueeze(0),
                                mode='bilinear',
                                padding_mode='zeros',
                                align_corners=True).squeeze().reshape(batch_size,patch_size,patch_size,patch_size)

        
        x_fbp  = x_fbp/len(angles[0])

        if self.output_patch_size is not None:
            x_fbp = x_fbp[:,patch_size//2-self.output_patch_size//2:patch_size//2+self.output_patch_size//2 + 1,
                            patch_size//2-self.output_patch_size//2:patch_size//2+self.output_patch_size//2 + 1,
                            patch_size//2-self.output_patch_size//2:patch_size//2+self.output_patch_size//2 + 1,]
    
        return x_fbp