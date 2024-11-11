"""
FFT Wrappers to torch.fft.fft to reproduce same results as pycbc/gwpy
(See e.g. the fft() method in gwpy.timeseries.TimeSeries class)
"""

import torch

__all__ = ['fft', 'rfft', 'fftfreq', 'rfftfreq', 'ifft', 'irfft']

class FFT():
    def __call__(self, input, norm=1, **kwargs):
        ft = torch.fft.fft(input, **kwargs) / norm
        #ft[...,1:]/=2
        return ft

class RFFT():
    def __call__(self, input, norm=1, **kwargs):
        ft = torch.fft.rfft(input, **kwargs) / norm
        #ft[...,1:]*=2
        return ft
    
class FFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.fftfreq(input, **kwargs)
    
class RFFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.rfftfreq(input, **kwargs)
    
class IFFT():
    def __call__(self, input, norm=1, **kwargs):
        #input[...,1:]/=2
        ift = torch.fft.ifft(input, **kwargs) * norm
        return ift
    
class IRFFT():
    def __call__(self, input, norm=1, **kwargs):
        #input[...,1:]/=2
        ift = torch.fft.irfft(input, **kwargs) * norm
        return ift
    

fft   = FFT()
rfft  = RFFT()
ifft  = IFFT()
irfft = IRFFT()
fftfreq  = FFTFREQ()
rfftfreq = RFFTFREQ()
