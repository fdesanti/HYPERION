"""
Transformer Embedding 
"""

import torch
import torch.nn as nn

from  hyperion.core.neural_networks.torch_layers import (GlobalAvgPooling1D, 
                                                         GlobalMaxPooling1D, 
                                                         Slicer)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerEmbedding(nn.Module):

    def __init__(self, 
                 strain_shape: list,
                 fs          : int,
                 use_batch_norm      = False,
                 activation          = nn.ELU(),
                 dropout_probability = 0.0,
                 CNN_filters         = [32, 64, 128],
                 CNN_kernel_sizes    = [1, 5, 5],
                 **kwargs,
                 ):
        
        super(TransformerEmbedding, self).__init__()

        self.strain_channels, self.strain_length = strain_shape
        
        #slicer kwargs
        overlap = kwargs.get('overlap', 15.0)
        segment_len = kwargs.get('segment_len', 0.1)
        
        self.slicer = Slicer(input_len=self.strain_length, fs=fs, segment_len=segment_len, overlap=overlap)

        
        #batch norm kwargs
        track_running_stats = kwargs.get('track_running_stats', True)


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


        #=======================================================================
        # Construct the Transformer Encoder
        #=======================================================================
        dim_feedforward    = kwargs.get('dim_feedforward', 2048)
        encoder_activation = kwargs.get('encoder_activation', 'relu')
        num_heads          = kwargs.get('num_heads', 32)
        num_encoder_layers = kwargs.get('num_encoder_layers', 2)

        self.encoder_layer = TransformerEncoderLayer(d_model         = CNN_filters[-1], 
                                                     nhead           = num_heads,
                                                     dim_feedforward = dim_feedforward,
                                                     dropout         = dropout_probability,
                                                     activation      = encoder_activation,
                                                     batch_first     = True)
        
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, 
                                                      num_layers = num_encoder_layers,
                                                      norm       = nn.LayerNorm(CNN_filters[-1]) if use_batch_norm else None)

        #final pooling layer ==============================================================================
        final_pooling = kwargs.get('final_pooling', 'avg')

        if final_pooling == 'avg':
            self.final_pooling = GlobalAvgPooling1D(data_format='channel_last')
        elif final_pooling == 'max':
            self.final_pooling = GlobalMaxPooling1D(data_format='channel_last')
        else:
            raise ValueError(f"Invalid final pooling method {final_pooling}. Choose between 'avg' and 'max'")

    @property
    def num_segments(self):
        return self.slicer.num_segments
    
    @property
    def segment_len(self):
        return self.slicer.segment_len
        
    
    def _convolutional_embedding(self, strain):
        """
        Embeds a multi-channel strain tensor with convolutional layers to be passed to the transformer
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

        return x_out
    
    def forward(self, strain, asd=None):
        """
        Forward pass of the Transformer Embedding
        """
        x = self._convolutional_embedding(strain)
        x = self.transformer_encoder(x)
        x = self.final_pooling(x)
        
        return x