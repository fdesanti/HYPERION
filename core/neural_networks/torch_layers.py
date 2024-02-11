"""
Implementation of some useful layers used in the network
"""
import torch
import torch.nn as nn


class GlobalMaxPooling1D(nn.Module):
     """Pytorch implementation of GlobalMaxPooling1D"""
     def __init__(self, data_format='channels_last'):
          super(GlobalMaxPooling1D, self).__init__()
          self.data_format = data_format
          self.step_axis = 1 if self.data_format == 'channels_last' else 2

     def forward(self, input):
          return torch.max(input, axis=self.step_axis).values
     
class GlobalAvgPooling1D(nn.Module):
     """Pytorch implementation of GlobalAvgPooling1D"""

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
