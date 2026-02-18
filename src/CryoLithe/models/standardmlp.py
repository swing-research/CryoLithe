"""
Standard mlp network
"""

from typing import Union, List
import torch
import torch.nn as nn



class standardMLP(nn.Module):
    """
     Standard MLP network Note: learn_residual is a dummy variable
     TODO: Correct the learn_residual variable
    """
    def __init__(self, input_size : int,
                 output_size : int, 
                 mlp_hidden : Union[int , List[int]], 
                 mlp_layers : int, 
                 batch_norm : bool = False, 
                 dropout=0.0, 
                 learn_residual=False,
                 skip_connection=False,
                 bias = True):
        super(standardMLP, self).__init__()

        self.input_size = input_size
        self.mlp_hidden = mlp_hidden
        self.mlp_layers = mlp_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.learn_residual = learn_residual
        self.skip_connection = skip_connection
        self.non_linearity = nn.ReLU()
        
        self.layers = nn.ModuleList()
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
        if self.dropout > 0:
            self.dropouts = nn.ModuleList()
        if isinstance(mlp_hidden, int):
            for i in range(self.mlp_layers):
                if i == 0:
                    self.layers.append(nn.Linear(self.input_size, self.mlp_hidden, bias=bias))
                else:
                    self.layers.append(nn.Linear(self.mlp_hidden, self.mlp_hidden, bias=bias))
                if self.batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(self.mlp_hidden))
                if self.dropout > 0:
                    self.dropouts.append(nn.Dropout(self.dropout))
            self.last_layer = nn.Linear(self.mlp_hidden, output_size, bias=bias)
        else:
            for i in range(mlp_layers):
                if i == 0:
                    self.layers.append(nn.Linear(self.input_size, mlp_hidden[0], bias=bias))
                else:
                    self.layers.append(nn.Linear(mlp_hidden[i-1], mlp_hidden[i], bias=bias))

                if self.batch_norm:
                    self.batch_norms.append(nn.BatchNorm1d(mlp_hidden[i]))
                if self.dropout > 0:
                    self.dropouts.append(nn.Dropout(self.dropout))
            self.last_layer = nn.Linear(mlp_hidden[-1], output_size, bias=bias)            
            
        

    
        
    def forward(self, x):
        """
        Forward pass of the network
        
        """
        #TODO: check skip connection
        for i in range(self.mlp_layers):
            if self.skip_connection:
                skip_input = x.clone()
            x = self.layers[i](x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = self.non_linearity(x)
            if self.dropout > 0:
                x = self.dropouts[i](x)

            if self.skip_connection and i>0:
                x = x + skip_input
        x = self.last_layer(x)
        return x