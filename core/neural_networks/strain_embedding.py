"""
Fully Connected Residual Neural Network implementation 
"""

import torch
import torch.nn as nn


from .torch_layers import GlobalMaxPooling1D


class ResBlock(nn.Module):
    """Implementation of the ResNet Block"""
    
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 use_batch_norm = True, 
                 activation = nn.ELU(),
                 dropout_probability=0.0):
        super(ResBlock, self).__init__()
        
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.use_batch_norm = use_batch_norm
        self.activation     = activation
        self.dropout        = nn.Dropout(dropout_probability)
        
        if use_batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(output_dim, track_running_stats=False)
            
        if input_dim != output_dim:
            self.linear_layer = nn.Linear(input_dim, output_dim)
            
        self.res_block = nn.ModuleList([
            nn.Linear(output_dim, output_dim)
            for _ in range(2)
        ])
        
    def forward(self, input):
        
        if self.input_dim != self.output_dim:
            x = self.activation(self.linear_layer(input))
        else:
            x = input
            
        residual = x 
        for layer in self.res_block:
            x = self.activation(layer(x))
            x = self.dropout(x)

        out = x + residual
        
        if self.use_batch_norm:
            out = self.batch_norm_layer(out)
            
        return out
    
    
class EmbeddingNetwork(nn.Module):
    
    def __init__(self, 
                 strain_shape : list, 
                 num_blocks   = 3,
                 block_dims :list = [2048, 1024, 512, 256], 
                 strain_out_dim = 256,
                 use_batch_norm = True, 
                 activation     = nn.ELU(),
                 dropout_probability=0.0):

        super(EmbeddingNetwork, self).__init__()
        
        self.strain_channels, self.strain_length = strain_shape
        
        self.num_blocks = num_blocks
        self.block_dims = block_dims
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        self.Dropout = nn.Dropout(dropout_probability)

        if use_batch_norm:
            self.initial_batch_norm = nn.BatchNorm1d(self.strain_channels, track_running_stats=False)
        
        #=======================================================================
        # Construct CNN for morphology features extraction
        #=======================================================================
        self.CNN = nn.Sequential(
                 
                 nn.Conv1d(self.strain_channels, 32, kernel_size = 13, stride = 1, bias = True),
                 nn.BatchNorm1d(32, track_running_stats=False) if use_batch_norm else nn.Identity(),
                 nn.MaxPool1d(2),
                 activation, 
                 
                 nn.Conv1d(32, 64, kernel_size = 7, stride = 1, bias = True),
                 nn.BatchNorm1d(64, track_running_stats=False) if use_batch_norm else nn.Identity(),
                 nn.MaxPool1d(2),
                 activation, 
            
                 nn.Conv1d(64, 128, kernel_size = 5, stride = 1, bias = True),
                 nn.BatchNorm1d(128, track_running_stats=False) if use_batch_norm else nn.Identity(),
                 nn.MaxPool1d(2),
                 activation,
                 
                 nn.Dropout(dropout_probability),
                 nn.Flatten(),
                 nn.LazyLinear(block_dims[0]), activation
                 )
        
        #=======================================================================
        # Construct CNN for time/space localization features extraction
        #=======================================================================
        filters      = [16, 32, 16, 32, 64, 128]
        #kernel_sizes = [7, 7, 5, 5, 3, 3]
        kernel_sizes = [128, 64, 32, 16, 8, 4]
            
        self.CNN_localization = nn.ModuleList(
             [
              nn.Sequential(
                          nn.Conv1d(strain_shape[0], filter, kernel_size = kernel_size, stride = 1, bias = True), 
                          nn.BatchNorm1d(filter, track_running_stats=False) if use_batch_norm else nn.Identity(),
                          self.activation, 
                          GlobalMaxPooling1D(),
              ) 
              for filter, kernel_size in zip(filters, kernel_sizes)    
             ] 
        )
        self.out_CNN_localization_block_linear = nn.Sequential(nn.LazyLinear(block_dims[0]), activation)
        

        #=======================================================================
        # Construct ResNet blocks
        #=======================================================================
        self.residual_blocks = []
        input_dim = block_dims[0]
        for block_dim in block_dims:
            for _ in range(num_blocks):
                self.residual_blocks.append( ResBlock(input_dim=input_dim, 
                                                      output_dim=block_dim, 
                                                      use_batch_norm=False, 
                                                      activation=activation,
                                                      dropout_probability= dropout_probability) )
                input_dim = block_dim
            
        self.resnet = nn.ModuleList(self.residual_blocks)
        
        #pre ResNet Layer
        self.linear = nn.LazyLinear(block_dims[0])

        #self.final_linear = nn.LazyLinear(strain_out_dim)
        '''
        self.ASD_embedding = nn.Sequential(
            nn.Flatten(), 
            nn.LazyLinear(4096), nn.ELU(),
            nn.LazyLinear(2048), nn.ELU(),
            nn.LazyLinear(1024), nn.ELU(),
        )
        '''
    def forward(self, strain, asd=None):
        
        #initial batch norm layer
        if self.use_batch_norm:
            s = self.initial_batch_norm(strain)
        else:
            s = strain
            
        #ASD Embedding
        #asd_embed = self.ASD_embedding(asd*1e20)
        

        #Morphology CNN
        out_CNN = self.CNN(s)
        #print('morphology CNN', out_CNN)

        
        #Localization CNN
        out_CNN_localization = torch.tensor([], device=strain.device)
        
        for cnn_layer in self.CNN_localization:
            out_CNN_localization = torch.cat([out_CNN_localization, cnn_layer(s)], dim = -1)
        out_CNN_localization = self.out_CNN_localization_block_linear(out_CNN_localization)
        
        out_CNN_localization = self.Dropout(out_CNN_localization)
        #print('out CNN localization', out_CNN_localization)
        
        #Combination of the two CNN blocks
        out = torch.cat([out_CNN, out_CNN_localization], dim = 1)
        
        out = self.activation(self.linear(out))
        
        #Final Resnet
        for res_layer in self.resnet:
            out = res_layer(out)

        #out = self.activation(self.final_linear(out))
        return out
            

        
        
        
