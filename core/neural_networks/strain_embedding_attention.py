"""
Fully Connected Residual Neural Network implementation with the Attention Mechanism
"""

import torch
import torch.nn as nn
from .strain_embedding import ResBlock


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
            
        return torch.stack(segments, dim = -2) # Has shape either (batch_size, num_strain_chanels, num_segments, segment_len)
                                               #             or   (batch_size, num_segments, segment_len)
       

    




class EmbeddingNetworkAttention(nn.Module):

    def __init__(self, 
                 strain_shape : list, 
                 num_blocks   = 3,
                 block_dims :list = [2048, 1024, 512, 256], 
                 strain_out_dim = 256,
                 use_batch_norm = True, 
                 activation     = nn.ELU(),
                 dropout_probability=0.0, 
                 CNN_filters      = [32, 64, 128], 
                 CNN_kernel_sizes = [5, 5, 5],
                 ):
        
        super(EmbeddingNetworkAttention, self).__init__()
        
        self.strain_channels, self.strain_length = strain_shape
        

    
    def _single_channel_embedding(self, x):
        """
        Embeds a single channel of the strain tensor
        """
        
        x = self.CNN(x)
        x = x.view(x.size(0), -1)
        x = self.FC(x)
        
        return x
    
    def forward(self, x):
        return