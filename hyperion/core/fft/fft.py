"""
FFT Wrappers to torch.fft.fft to reproduce same results as pycbc/gwpy
(See e.g. the fft() method in gwpy.timeseries.TimeSeries class)
"""

import torch

class FFT():
    """
    Wrapper to torch.fft.fft to reproduce same results as pycbc/gwpy
    (See e.g. the fft() method in gwpy.timeseries.TimeSeries class)

    Args:
        input (torch.Tensor): Input tensor
        norm (int, optional): Normalization factor. (Default 1)
        kwargs              : Additional arguments to torch.fft.fft
    """
    def __call__(self, input, norm=1, **kwargs):
        ft = torch.fft.fft(input, **kwargs) / norm
        return ft

class RFFT():
    """
    Wrapper to torch.fft.rfft to reproduce same results as pycbc/gwpy
    (See e.g. the fft() method in gwpy.timeseries.TimeSeries class)

    Args:
        input (torch.Tensor): Input tensor
        norm (int, optional): Normalization factor. (Default 1)
        kwargs              : Additional arguments to torch.fft.rfft
    """
    def __call__(self, input, norm=1, **kwargs):
        ft = torch.fft.rfft(input, **kwargs) / norm
        ft[..., 1:] *= 2.0
        return ft
    
class FFTFREQ():
    """
    Wrapper to torch.fft.fftfreq to reproduce same results as pycbc/gwpy

    Args:
        input (torch.Tensor): Input tensor
        kwargs              : Additional arguments to torch.fft.fftfreq
    """
    def __call__(self, input, **kwargs):
        return torch.fft.fftfreq(input, **kwargs)
    
class RFFTFREQ():
    """
    Wrapper to torch.fft.rfftfreq to reproduce same results as pycbc/gwpy

    Args:
        input (torch.Tensor): Input tensor
        kwargs              : Additional arguments to torch.fft.rfftfreq
    """
    def __call__(self, input, **kwargs):
        return torch.fft.rfftfreq(input, **kwargs)
    
class IFFT():
    """
    Wrapper to torch.fft.ifft to reproduce same results as pycbc/gwpy
    (See e.g. the fft() method in gwpy.timeseries.TimeSeries class)

    Args:
        input (torch.Tensor): Input tensor
        norm (int, optional): Normalization factor. (Default 1)
        kwargs              : Additional arguments to torch.fft.ifft
    """
    def __call__(self, input, norm=1, **kwargs):
        ift = torch.fft.ifft(input*norm, **kwargs) 
        return ift
    
class IRFFT():
    """
    Wrapper to torch.fft.irfft to reproduce same results as pycbc/gwpy
    (See e.g. the fft() method in gwpy.timeseries.TimeSeries class)

    Args:
        input (torch.Tensor): Input tensor
        norm (int, optional): Normalization factor. (Default 1)
        kwargs              : Additional arguments to torch.fft.irfft
    """
    def __call__(self, input, norm=1, **kwargs):
        ft = input*norm
        ft[..., 1:] /= 2.0
        ift = torch.fft.irfft(ft, **kwargs) 
        return ift
    
fft      = FFT()
rfft     = RFFT()
ifft     = IFFT()
irfft    = IRFFT()
fftfreq  = FFTFREQ()
rfftfreq = RFFTFREQ()
