"""
Implementation of some useful layers used in the network
"""
import torch
import torch.nn as nn


class GlobalMaxPooling1D(nn.Module):
     """
     Pytorch implementation of GlobalMaxPooling1D
     
     Args:
        data_format (str): The data format. Can be 'channels_last' or 'channels_first
     """
     def __init__(self, data_format='channels_last'):
          super(GlobalMaxPooling1D, self).__init__()
          self.data_format = data_format
          self.step_axis = 1 if self.data_format == 'channels_last' else 2

     def forward(self, input):
          return torch.max(input, axis=self.step_axis).values
     
class GlobalAvgPooling1D(nn.Module):
     """
     Pytorch implementation of GlobalAvgPooling1D
     
     Args:
        data_format (str): The data format. Can be 'channels_last' or 'channels_first
     """
     def __init__(self, data_format='channels_last'):
          super(GlobalAvgPooling1D, self).__init__()
          self.data_format = data_format
          self.step_axis = 1 if self.data_format == 'channels_last' else 2

     def forward(self, input):
          return torch.mean(input, axis=self.step_axis)

class SeparableConv1d(nn.Module):
    """
    Implements SeparableConv1d as in the Xception architecture

    It works with inputs x = [Batch_size, in_channels, in_length]
    The initial datas must be of this form x = [Batch_size, N_detectors, len_strain]

    It should be faster than standard convolutions

    Args:
        in_channels  (int)   : The number of input channels
        out_channels (int)   : The number of output channels
        kernel_size  (int)   : The size of the kernel
        bias         (bool)  : If True, adds a learnable bias to the output
        stride       (int)   : The stride of the convolution
        padding      (int)   : The padding of the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias, stride, padding = 0):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, bias=bias, padding = padding, )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 
                                kernel_size=1, bias=bias, padding = padding, stride = stride)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

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
            self.batch_norm_layer = nn.LayerNorm(output_dim)#nn.BatchNorm1d(output_dim, track_running_stats=False)
            
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
    
class Slicer(nn.Module):
    """
    Slices the input strain tensor into 
    (possibly overlapping) windows of a fixed length

    Args:
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

    def unslice(self, segments):
        """
        segments: (batch, channels, num_segments, segment_len)
        returns: reconstructed (batch, channels, input_len)
        """
        B, C, N, L = segments.shape
        device = segments.device

        # Prepare output buffers
        output = torch.zeros(B, C, self.input_len, device=device)
        count  = torch.zeros_like(output)

        for i in range(N):
            start = i * self.step
            output[..., start:start+L] += segments[..., i, :]
            count[...,  start:start+L] += 1

        # avoid division by zero (shouldn't happen if parameters are consistent)
        count = torch.where(count == 0, torch.ones_like(count), count)
        return output / count