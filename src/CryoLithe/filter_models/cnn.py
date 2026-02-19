import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self,
              fitler_size=3, 
              hidden_channels = 3, 
              hidden_layers= 3,
              padding_mode = 'reflect',
              nonlinearity = nn.LeakyReLU()):
        super(CNN, self).__init__()
        self.fitler_size = fitler_size
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.padding_mode = padding_mode
        self.non_linearity = nonlinearity
        #self.residual = residual
        self.convInput = torch.nn.Conv2d(1,self.hidden_channels,self.fitler_size,padding='same', padding_mode=self.padding_mode)
        self.convHidden = nn.ModuleList()
        for i in range(self.hidden_layers):
            self.convHidden.append(torch.nn.Conv2d(self.hidden_channels ,
                                                   self.hidden_channels ,
                                                   self.fitler_size,
                                                   padding='same', padding_mode=self.padding_mode))

        self.convOutput =  torch.nn.Conv2d(self.hidden_channels,1,self.fitler_size,padding='same', padding_mode=self.padding_mode)
        
    def forward(self,x):
        """
        x = N_projectionxNxN
        """
        input = x[:,None].clone()
        x = self.convInput(x[:,None])

        for i in range(self.hidden_layers):
            x = self.convHidden[i](x)
            x = self.non_linearity(x)

        x = self.convOutput(x)
        #if self.residual:
       #     x = x + input
        return  x.squeeze()

# class CNN(torch.nn.Module):
#     def __init__(self,
#               fitler_size=3, 
#               hidden_channels = 3, 
#               hidden_layers= 3,
#               padding_mode = 'reflect',
#               nonlinearity = nn.LeakyReLU(),
#               residual  = False):
#         super(CNN, self).__init__()
#         self.fitler_size = fitler_size
#         self.hidden_channels = hidden_channels
#         self.hidden_layers = hidden_layers
#         self.padding_mode = padding_mode
#         self.non_linearity = nonlinearity
#         self.residual = residual
#         self.convInput = torch.nn.Conv2d(1,self.hidden_channels,self.fitler_size,padding='same', padding_mode=self.padding_mode)
#         self.convHidden = nn.ModuleList()
#         for i in range(self.hidden_layers):
#             self.convHidden.append(torch.nn.Conv2d(self.hidden_channels ,
#                                                    self.hidden_channels ,
#                                                    self.fitler_size,
#                                                    padding='same', padding_mode=self.padding_mode))

#         self.convOutput =  torch.nn.Conv2d(self.hidden_channels,1,self.fitler_size,padding='same', padding_mode=self.padding_mode)
        
#     def forward(self,x):
#         """
#         x = N_projectionxNxN
#         """
#         input = x[:,None].clone()
#         x = self.convInput(x[:,None])

#         for i in range(self.hidden_layers):
#             x = self.convHidden[i](x)
#             x = self.non_linearity(x)

#         x = self.convOutput(x)
#         if self.residual:
#             x = x + input
#         return  x.squeeze()