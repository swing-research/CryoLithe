"""
slice mlp model for tomography reconstruction
"""
from typing import Union, List
import torch
import torch.nn as nn

from models.standardmlp import standardMLP


class sliceMLP(nn.Module):
    """
    MLP along the slices of the projection and an mlp to combine the slices
    """
    def __init__(self, n_projections: int,
                 mlp_output: int,
                 patch_size: int,
                 mlp_layers : int,
                 mlp_hidden : int,
                 dropout : int = 0,
                 batch_norm : bool = False,
                 learn_residual : bool = False,
                 skip_connection : bool = False,
                 slice_index: int = 2,
                 compare_index: int = 3):
        super(sliceMLP, self).__init__()
        #TODO: Add skip connection

        self.n_projections = n_projections
        self.mlp_output = mlp_output
        self.patch_size = patch_size
        self.dropout = dropout
        self.mlp_layers = mlp_layers
        self.mlp_hidden = mlp_hidden
        self.slice_index = slice_index
        self.compare_index = compare_index
        self.batch_norm = batch_norm
        self.learn_residual = learn_residual
        self.skip_connection = skip_connection

        self.slice_mlp= standardMLP(input_size = self.patch_size*self.n_projections,
                                          output_size = self.mlp_output,
                                          mlp_hidden = self.mlp_hidden,
                                          mlp_layers = self.mlp_layers,
                                          batch_norm = self.batch_norm,
                                          dropout = self.dropout,
                                          learn_residual = self.learn_residual,
                                          skip_connection = self.skip_connection)

        self.combination_mlp = standardMLP(input_size = self.patch_size*self.mlp_output,
                                           output_size = 1,
                                           mlp_hidden = self.mlp_hidden,
                                           mlp_layers = self.mlp_layers,
                                           batch_norm = self.batch_norm,
                                           dropout = self.dropout,
                                           learn_residual=self.learn_residual,
                                           skip_connection = self.skip_connection)
        

    def forward(self, x):
        """
        Forward pass of the network
        """
        if self.learn_residual:
            mid_pix = x[:,:,self.patch_size//2,self.patch_size//2]
            mid_pix_sum = torch.mean(mid_pix,1)
        x = x.permute(0,self.slice_index,1,self.compare_index).contiguous()        
        x = x.reshape(-1,self.patch_size*self.n_projections)
        x = self.slice_mlp(x).reshape(-1,self.patch_size,self.mlp_output).reshape(-1,self.patch_size*self.mlp_output)

        x = self.combination_mlp(x)
        if self.learn_residual:
            x = x + mid_pix_sum.unsqueeze(1)
        return x
    

class sliceMlp_v2(nn.Module):
    """
    MLP along the slices of the projection and an mlp to combine the slices with a different architecture for
    encoding and combining
    """
    def __init__(self, n_projections: int,
                 mlp_output: int,
                 patch_size: int,
                 slice_index: int = 2,
                 compare_index: int = 3,
                 learn_residual: bool = False,
                 slice_mlp_layers : int = 5,
                 slice_mlp_hidden : int = 256,
                 slice_dropout : int = 0,
                 slice_batch_norm : bool = False,
                 slice_learn_residual : bool = False,
                 slice_skip_connection : bool = False,
                 combine_mlp_layers : int = 5,
                 combine_mlp_hidden : int = 256,
                 combine_dropout : int = 0,
                 combine_batch_norm : bool = False,
                 combine_learn_residual : bool = False,
                 combine_skip_connection : bool = False,
                 output_size : int = 1):
        super(sliceMlp_v2, self).__init__()

        self.n_projections = n_projections
        self.mlp_output = mlp_output
        self.patch_size = patch_size
        self.slice_index = slice_index
        self.learn_residual = learn_residual 
        self.compare_index = compare_index
        
        self.slice_mlp= standardMLP(input_size = self.patch_size*self.n_projections,
                                          output_size = self.mlp_output,
                                          mlp_hidden = slice_mlp_hidden,
                                          mlp_layers = slice_mlp_layers,
                                          batch_norm = slice_batch_norm,
                                          dropout = slice_dropout,
                                          learn_residual = slice_learn_residual,
                                          skip_connection = slice_skip_connection)
        
        self.combination_mlp = standardMLP(input_size = self.patch_size*self.mlp_output,
                                           output_size = output_size,
                                           mlp_hidden = combine_mlp_hidden,
                                           mlp_layers = combine_mlp_layers,
                                           batch_norm = combine_batch_norm,
                                           dropout = combine_dropout,
                                           learn_residual=combine_learn_residual,
                                           skip_connection = combine_skip_connection)
        

    def forward(self, x):
        """
        Forward pass of the network
        """
        if self.learn_residual:
            mid_pix = x[:,:,self.patch_size//2,self.patch_size//2]
            mid_pix_sum = torch.mean(mid_pix,1)

        
        x = x.permute(0,self.slice_index,1,self.compare_index).contiguous() 
        x = x.reshape(-1,self.patch_size*self.n_projections)
        x = self.slice_mlp(x).reshape(-1,self.patch_size,self.mlp_output).reshape(-1,self.patch_size*self.mlp_output)

        x = self.combination_mlp(x)
        if self.learn_residual:
            x = x + mid_pix_sum.unsqueeze(1)
        return x
    
class sliceMlp_mulitnet(nn.Module):
    """
    Separate mlp for each slice of the patch and then combine the output of the mlp with another mlp
    """
    def __init__(self, n_projections: int,
                 mlp_output: int,
                 patch_size: int,
                 slice_index: int = 2,
                 compare_index: int = 3,
                 learn_residual: bool = False,
                 slice_mlp_layers : int = 5,
                 slice_mlp_hidden : int = 256,
                 slice_dropout : int = 0,
                 slice_batch_norm : bool = False,
                 slice_learn_residual : bool = False,
                 slice_skip_connection : bool = False,
                 slice_bias : bool = True,
                 combine_mlp_layers : int = 5,
                 combine_mlp_hidden : int = 256,
                 combine_dropout : int = 0,
                 combine_batch_norm : bool = False,
                 combine_learn_residual : bool = False,
                 combine_skip_connection : bool = False,
                 combine_bias : bool = True,
                 output_size : int = 1):
        super(sliceMlp_mulitnet, self).__init__()

        self.n_projections = n_projections
        self.mlp_output = mlp_output
        self.patch_size = patch_size
        self.slice_index = slice_index
        self.learn_residual = learn_residual 
        self.compare_index = compare_index
        self.slice_mlps = nn.ModuleList()

        for i in range(patch_size):
            self.slice_mlps.append(standardMLP(input_size = self.patch_size*self.n_projections,
                                          output_size = self.mlp_output,
                                          mlp_hidden = slice_mlp_hidden,
                                          mlp_layers = slice_mlp_layers,
                                          batch_norm = slice_batch_norm,
                                          dropout = slice_dropout,
                                          learn_residual = slice_learn_residual,
                                          skip_connection = slice_skip_connection,
                                          bias = slice_bias))
        
        self.combination_mlp = standardMLP(input_size = self.patch_size*self.mlp_output,
                                           output_size = output_size,
                                           mlp_hidden = combine_mlp_hidden,
                                           mlp_layers = combine_mlp_layers,
                                           batch_norm = combine_batch_norm,
                                           dropout = combine_dropout,
                                           learn_residual=combine_learn_residual,
                                           skip_connection = combine_skip_connection,
                                           bias = combine_bias)
        

    def forward(self, x):
        """
        Forward pass of the network
        """
        if self.learn_residual:
            mid_pix = x[:,:,self.patch_size//2,self.patch_size//2]
            mid_pix_sum = torch.mean(mid_pix,1)
        x = x.permute(0,self.slice_index,1,self.compare_index).contiguous() 
        x = x.reshape(-1,self.patch_size,self.patch_size*self.n_projections)
        x = [self.slice_mlps[i](x[:,i]) for i in range(self.patch_size)]
        x = torch.cat(x,1)
        x = self.combination_mlp(x)
        if self.learn_residual:
            x = x + mid_pix_sum.unsqueeze(1)
        #print(x.shape)
        return x


class sliceMlp_multinet_multiproj(nn.Module):
    """
    Separate mlp for each slice of the patch and then combine the output of the mlp with another mlp
    """
    def __init__(self, n_projections: int,
                 mlp_output: int,
                 patch_size: int,
                 slice_index: int = 2,
                 compare_index: int = 3,
                 n_series: int = 2,
                 learn_residual: bool = False,
                 slice_mlp_layers : int = 5,
                 slice_mlp_hidden : int = 256,
                 slice_dropout : int = 0,
                 slice_batch_norm : bool = False,
                 slice_learn_residual : bool = False,
                 slice_skip_connection : bool = False,
                 combine_mlp_layers : int = 5,
                 combine_mlp_hidden : int = 256,
                 combine_dropout : int = 0,
                 combine_batch_norm : bool = False,
                 combine_learn_residual : bool = False,
                 combine_skip_connection : bool = False,
                 output_size : int = 1):
        super(sliceMlp_multinet_multiproj, self).__init__()

        self.n_projections = n_projections
        self.mlp_output = mlp_output
        self.patch_size = patch_size
        self.slice_index = slice_index
        self.learn_residual = learn_residual 
        self.compare_index = compare_index
        self.n_series = n_series
        self.slice_mlps = nn.ModuleList()

        for i in range(patch_size*n_series):
            self.slice_mlps.append(standardMLP(input_size = self.patch_size*self.n_projections,
                                          output_size = self.mlp_output,
                                          mlp_hidden = slice_mlp_hidden,
                                          mlp_layers = slice_mlp_layers,
                                          batch_norm = slice_batch_norm,
                                          dropout = slice_dropout,
                                          learn_residual = slice_learn_residual,
                                          skip_connection = slice_skip_connection))
        
        self.combination_mlp = standardMLP(input_size = self.patch_size*self.mlp_output*self.n_series,
                                           output_size = output_size,
                                           mlp_hidden = combine_mlp_hidden,
                                           mlp_layers = combine_mlp_layers,
                                           batch_norm = combine_batch_norm,
                                           dropout = combine_dropout,
                                           learn_residual=combine_learn_residual,
                                           skip_connection = combine_skip_connection)
        

    def forward(self, x):
        """
        Forward pass of the network
        """
        if self.learn_residual:
            mid_pix = x[:,:self.n_projections,self.patch_size//2,self.patch_size//2]
            mid_pix_sum = torch.mean(mid_pix,1)

            
        x = x.permute(0,self.slice_index,1,self.compare_index).contiguous() 
        x_op = []
        for i in range(self.n_series):
            x_sub = x[:,:,i*(self.n_projections):(i+1)*(self.n_projections)]
            x_sub = x_sub.reshape(-1,self.patch_size,self.patch_size*self.n_projections)
            x_sub = [self.slice_mlps[i](x_sub[:,i]) for i in range(self.patch_size)]
            x_sub = torch.cat(x_sub,1)
            x_op.append(x_sub)
        x_op = torch.cat(x_op,1)
        x_op = self.combination_mlp(x_op)
        if self.learn_residual:
            x_op = x_op + mid_pix_sum.unsqueeze(1)
        #print(x.shape)
        return x_op
    
    


