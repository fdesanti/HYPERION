"""
Fully Connected Residual Neural Network implementation with the Attention Mechanism
"""

import torch
import torch.nn as nn
from  hyperion.core.neural_networks.strain_embedding import ResBlock
from  hyperion.core.neural_networks.torch_layers import GlobalMaxPooling1D, Slicer


class EmbeddingNetworkAttention(nn.Module):

    def __init__(self, 
                 strain_shape: list,
                 fs          : int,
                 num_blocks          = 3,
                 block_dims  :list   = [2048, 1024, 512, 256],
                 strain_out_dim      = 256,
                 use_batch_norm      = False,
                 activation          = nn.ELU(),
                 dropout_probability = 0.0,
                 CNN_filters         = [32, 64, 128],
                 CNN_kernel_sizes    = [1, 5, 5],
                 **kwargs,
                 ):
        
        super(EmbeddingNetworkAttention, self).__init__()
        
        #slicer kwargs
        overlap = kwargs.get('overlap', 15.0)
        segment_len = kwargs.get('segment_len', 0.1)
        
        #attention kwargs
        num_heads   = kwargs.get('num_heads', 32)
        add_bias_kv = kwargs.get('add_bias_kv', False)
        
        #batch norm kwargs
        track_running_stats = kwargs.get('track_running_stats', True)


        self.strain_channels, self.strain_length = strain_shape
        self.slicer = Slicer(input_len=self.strain_length, fs=fs, segment_len=segment_len, overlap=overlap)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.initial_batch_norm = nn.BatchNorm1d(self.strain_channels, track_running_stats=track_running_stats)

        #=======================================================================
        # Construct CNN for morphology features extraction
        #=======================================================================
        shapes = [self.strain_channels] + CNN_filters
        CNN_layers = [nn.Sequential(
                      nn.Conv1d(in_channel, filter, kernel_size = kernel_size, stride = 1, bias = True),
                      nn.BatchNorm1d(filter, track_running_stats=False) if use_batch_norm else nn.Identity(),
                      nn.Dropout(dropout_probability),
                      activation,
                      )
                      for in_channel, filter, kernel_size in zip(shapes, CNN_filters, CNN_kernel_sizes)
                ]           
        self.CNN = nn.Sequential(*CNN_layers)

        #Multihead Attention Layer
        self.MHA = nn.MultiheadAttention(embed_dim   = CNN_filters[-1], 
                                         num_heads   = num_heads,
                                         add_bias_kv = add_bias_kv,
                                         dropout     = dropout_probability,
                                         batch_first = True)

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
        self.pre_resnet_linear = nn.Sequential(nn.LazyLinear(block_dims[0]), activation)


    @property
    def num_segments(self):
        return self.slicer.num_segments
    
    @property
    def segment_len(self):
        return self.slicer.segment_len
        
    
    def _strain_embedding(self, strain):
        """
        Embeds a multi-channel strain tensor with convolutional layers & Attention Mechanism
        """
        # Slice the input tensor into segments
        sliced_strain = self.slicer(strain) # (batch_size, 1, num_segments, segment_len)
        
        x_out = []
        for i in range(self.num_segments):
            x_i = sliced_strain[:, :, i, :]
            x_i = self.CNN(x_i)
            x_i = GlobalMaxPooling1D(data_format='channel_last')(x_i)
            x_out.append(x_i)
        x_out = torch.stack(x_out, dim=1) # (batch_size, num_segments, CNN_filters[-1])
        
        #compute the attention 
        attn, _ = self.MHA(x_out, x_out, x_out, need_weights=False)
        
        #return the flattened attention
        return nn.Flatten()(attn)
    
    def forward(self, strain, asd = None):

        if self.use_batch_norm:
            strain = self.initial_batch_norm(strain)
        '''
        embedded_strain = torch.stack([self._single_detector_embedding(strain[:, i, ...]) 
                                for i in range(self.strain_channels)], dim=-1).sum(dim=-1)
        '''
        
        embedded_strain = self._strain_embedding(strain)
        
        embedded_strain = self.pre_resnet_linear(embedded_strain)

        #Final Resnet
        for res_layer in self.resnet:
            embedded_strain = res_layer(embedded_strain)        
        
        return embedded_strain
    