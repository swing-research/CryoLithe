"""
Set model for  combining the informations from the patches
"""






from torch import nn
import torch
from models.standardmlp import standardMLP





# Acts along the slices
class RadonSet(nn.Module):
    def __init__(self, input = 21,
                 output = 1,
                 set_input =512,
                 set_output = None,
                 transformer_positional_encoding_size = 128,
                 use_learned_positional_encoding = False,
                 transformer_positional_encoding_base = 1000,
                 transformer_positional_encoding_add_angle : bool = False,
                 transformer_positional_encoding_mult_angle: bool = False,
                 set_hidden_size = 1024,
                 set_num_layers = 3,
                 set_skip_connection = False,
                 set_bias = False,
                 transformer_avg_pooling = True,
                 mlp_hidden_size=512,
                 mlp_num_layers=2,
                 mlp_skip_connection=False,
                 mlp_bias = False,
                 bias = True):
        
        super(RadonSet, self).__init__()

        self.up_layer = nn.Linear(input, set_input, bias=bias)
        self.transformer_positional_encoding_size = transformer_positional_encoding_size
        self.transformer_positional_encoding_base = transformer_positional_encoding_base
        self.transformer_positional_encoding_add_angle = transformer_positional_encoding_add_angle
        self.transformer_avg_pooling = transformer_avg_pooling
        self.use_learned_positional_encoding = use_learned_positional_encoding
        self.transformer_positional_encoding_mult_angle = transformer_positional_encoding_mult_angle
        if set_output is None:
            set_output = set_input


        self.setMlP = standardMLP(input_size = set_input,
                                    output_size = set_output,
                                    mlp_hidden = set_hidden_size,
                                    mlp_layers = set_num_layers,
                                    skip_connection = set_skip_connection,
                                    bias = set_bias)



        self.mlp = standardMLP(input_size = set_output,
                               output_size = output,
                               mlp_hidden = mlp_hidden_size,
                               mlp_layers = mlp_num_layers,
                               skip_connection = mlp_skip_connection,
                               bias = mlp_bias) 

    def forward(self, x, angle):
        x = self.up_layer(x)
        if self.transformer_positional_encoding_add_angle:
            x = x + angle
        elif self.transformer_positional_encoding_mult_angle:
            x = x*angle
        else:
            #print('concat angle')
            x = torch.cat((x,angle),2)

        x = self.setMlP(x)

        # Average pooling
        if self.transformer_avg_pooling:
            x = torch.mean(x, dim=1)
        else:
            x = x[:,0]
        x = self.mlp(x)
        return x
    


class SliceSet(nn.Module):
    """
    Separate transformer for each slice and the combine the output
    """
    def __init__(self, n_projections: int,
                 mlp_output: int,
                 patch_size: int,
                 slice_index: int = 2,
                 compare_index: int = 3,
                 learn_residual: bool = False,
                 set_input : int = 512,
                 set_output : int = None,
                 set_hidden_size = 1024,
                 set_num_layers = 3,
                 set_skip_connection = False,
                 set_bias = False,
                 radon_bias = True,
                 learned_positional_encoding = False,
                 learned_positional_encoding_use_softmax = False,
                 slice_transformer_avg_pooling : bool = True,
                 slice_transformer_positional_encoding_size : int = 128,
                 slice_transformer_positional_encoding_base : int = 1000,
                 sice_transformer_transformer_positional_encoding_add_angle : bool = False,
                 sice_transformer_positional_encoding_mult_angle: bool = False,
                 slice_mlp_hidden_size : int = 512,
                 slice_mlp_num_layers : int = 3,
                 slice_mlp_skip_connection : bool = False,
                 slice_mlp_bias : bool = False,
                 combine_mlp_layers : int = 5,
                 combine_mlp_hidden : int = 256,
                 combine_dropout : int = 0,
                 combine_batch_norm : bool = False,
                 combine_learn_residual : bool = False,
                 combine_skip_connection : bool = False,
                 combine_mlp_bias : bool = False,
                 output_size : int = 1):
        super(SliceSet, self).__init__()

        self.n_projections = n_projections
        self.mlp_output = mlp_output
        self.patch_size = patch_size
        self.slice_index = slice_index
        self.learn_residual = learn_residual 
        self.compare_index = compare_index
        self.slice_transformers = nn.ModuleList()
        self.use_learned_positional_encoding = learned_positional_encoding
        self.slice_transformer_positional_encoding_size = slice_transformer_positional_encoding_size
        self.slice_transformer_positional_encoding_base = slice_transformer_positional_encoding_base
        self.learned_positional_encoding_use_softmax = learned_positional_encoding_use_softmax

        if self.learned_positional_encoding_use_softmax:
            self.pos_nonlinear = nn.Softmax(dim=-1)
        else:
            self.pos_nonlinear = nn.Identity()


        if self.use_learned_positional_encoding:
            self.pos_encoder = nn.Linear(1, slice_transformer_positional_encoding_size, bias=radon_bias)


        for i in range(patch_size):
            self.slice_transformers.append(RadonSet(input = self.patch_size,
                                                    output = self.mlp_output,
                                                    set_input = set_input,
                                                    set_output = set_output,
                                                    set_hidden_size = set_hidden_size,
                                                    set_num_layers = set_num_layers,
                                                    set_skip_connection = set_skip_connection,
                                                    set_bias = set_bias,
                                                    use_learned_positional_encoding = learned_positional_encoding,
                                                    transformer_positional_encoding_size = slice_transformer_positional_encoding_size,
                                                    transformer_positional_encoding_base = slice_transformer_positional_encoding_base,
                                                    transformer_positional_encoding_add_angle = sice_transformer_transformer_positional_encoding_add_angle,
                                                    transformer_positional_encoding_mult_angle = sice_transformer_positional_encoding_mult_angle,
                                                    mlp_hidden_size= slice_mlp_hidden_size,
                                                    mlp_num_layers= slice_mlp_num_layers,
                                                    mlp_skip_connection= slice_mlp_skip_connection,
                                                    mlp_bias = slice_mlp_bias,
                                                    transformer_avg_pooling = slice_transformer_avg_pooling,
                                                    bias = radon_bias))
        
        self.combination_mlp = standardMLP(input_size = self.patch_size*self.mlp_output,
                                           output_size = output_size,
                                           mlp_hidden = combine_mlp_hidden,
                                           mlp_layers = combine_mlp_layers,
                                           batch_norm = combine_batch_norm,
                                           dropout = combine_dropout,
                                           learn_residual=combine_learn_residual,
                                           skip_connection = combine_skip_connection,
                                           bias = combine_mlp_bias)
        

    def forward(self, x,angles):
        """
        Forward pass of the network
        """
        if self.learn_residual:
            mid_pix = x[:,:,self.patch_size//2,self.patch_size//2]
            mid_pix_sum = torch.mean(mid_pix,1)
        x = x.permute(0,self.slice_index,1,self.compare_index).contiguous() 

        if self.use_learned_positional_encoding:
            angles = self.pos_nonlinear(self.pos_encoder(angles.unsqueeze(-1)))
        else:
            angles = self.angle_encoding(angles)
        x = [self.slice_transformers[i](x[:,i],angles) for i in range(self.patch_size)]
        x = torch.cat(x,1)
        x = self.combination_mlp(x)
        if self.learn_residual:
            x = x + mid_pix_sum.unsqueeze(1)
        return x 

    def angle_encoding(self, angle):
        """
        Angle encoding using sin and cos
        angle: tensor of shape (batch_size, n_projections)
        output: tensor of shape (batch_size, n_projections, self.transformer_positional_encoding_size)
        """

        d = self.slice_transformer_positional_encoding_size
        base_frequency = self.slice_transformer_positional_encoding_base

        samples = angle.shape[0]

        pe  = torch.zeros(samples, angle.shape[1], d, device = angle.device, dtype = angle.dtype)

        for i in range(d//2):
            pe[:,:,2*i] = torch.sin(angle[0,:]*(180/torch.pi)/(base_frequency**(2*i/d)))
            pe[:,:,2*i+1] = torch.cos(angle[0,:]*(180/torch.pi)/(base_frequency**(2*i/d)))
    
        return pe   
    
