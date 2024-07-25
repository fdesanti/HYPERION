"""
Fully Connected Residual Neural Network implementation with the Attention Mechanism
"""

import torch
import torch.nn as nn
from hyperion.core.neural_networks.strain_embedding import ResBlock
from  hyperion.core.neural_networks.torch_layers import GlobalMaxPooling1D



class Slicer(nn.Module):
    """
    Slices the input strain tensor into 
    (possibly overlapping) windows of a fixed length

    Args:
    -----
        input_len (int)    : The length of the input strain tensor
        fs (int)           : The sampling frequency of the input strain tensor
        segment_len (float): The length in seconds of the output segments
        overlap (float)    : The overlap (percentage) between segments in seconds
    """
    
    def __init__(self, 
                 input_len, 
                 fs,
                 segment_len = 0.1,
                 overlap = 0.0):
        super(Slicer, self).__init__()
        
        self.fs = fs
        self.input_len = input_len
        self.segment_len = int(segment_len * fs) # Convert seconds to samples
        self.step = int(self.segment_len * (1 - overlap/100))

    @property
    def num_segments(self):
        return (self.input_len - self.segment_len) // self.step + 1
        
    def forward(self, x):
        """
        Slices the input tensor into segments of length segment_len
        """       
        segments = []
        for i in range(0, self.input_len - self.segment_len, self.step):
            segments.append(x[..., i:i+self.segment_len])
        
        segments = torch.stack(segments, dim = -2).unsqueeze(-1)

        return segments.view(x.shape[0], x.shape[1], self.num_segments, self.segment_len) # (batch_size, num_channels, num_segments, segment_len)
       

class EmbeddingNetworkAttention(nn.Module):

    def __init__(self, 
                 strain_shape : list, 
                 num_blocks   = 3,
                 block_dims :list = [2048, 1024, 512, 256], 
                 strain_out_dim = 256,
                 use_batch_norm = False, 
                 activation     = nn.ELU(),
                 dropout_probability=0.0, 
                 CNN_filters      = [32, 64, 128], 
                 CNN_kernel_sizes = [1, 5, 5],
                 num_heads        = 32,
                 **slicer_kwargs
                 ):
        
        super(EmbeddingNetworkAttention, self).__init__()
        
        fs = slicer_kwargs.get('fs', 2048)
        overlap = slicer_kwargs.get('overlap', 0.0)
        segment_len = slicer_kwargs.get('segment_len', 0.1)


        self.strain_channels, self.strain_length = strain_shape
        self.slicer = Slicer(input_len=self.strain_length, fs=fs, segment_len=segment_len, overlap=overlap)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.initial_batch_norm = nn.BatchNorm1d(self.strain_channels, track_running_stats=False)

        #=======================================================================
        # Construct CNN for morphology features extraction
        #=======================================================================
        shapes = [self.strain_channels] + CNN_filters
        CNN_layers = [nn.Sequential(
                      nn.Conv1d(in_channel, filter, kernel_size = kernel_size, stride = 1, bias = True),
                      nn.BatchNorm1d(filter, track_running_stats=False) if use_batch_norm else nn.Identity(),
                      activation,
                      )
                      for in_channel, filter, kernel_size in zip(shapes, CNN_filters, CNN_kernel_sizes)
                ]           
        self.CNN = nn.Sequential(*CNN_layers)

        #Multihead Attention Layer
        self.MHA = nn.MultiheadAttention(embed_dim=CNN_filters[-1], 
                                         num_heads=num_heads, 
                                         batch_first=True)

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
        
    
    def _single_detector_embedding(self, strain):
        """
        Embeds a single channel of the strain tensor
        """
        # Slice the input tensor into segments
        sliced_strain = self.slicer(strain) # (batch_size, 1, num_segments, segment_len)
        
        x_out = []
        for i in range(self.num_segments):
            x_i = sliced_strain[:, :, i, :]
            x_i = self.CNN(x_i)
            x_i = nn.Flatten()(x_i)#GlobalMaxPooling1D()(x_i)
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
        
        

        embedded_strain = self.pre_resnet_linear(embedded_strain)

        #Final Resnet
        for res_layer in self.resnet:
            embedded_strain = res_layer(embedded_strain)        
        
        return embedded_strain
    