"""FFT Wrappers to torch.fft.fft to reproduce same results as pycbc"""

import torch

__all__ = ['fft', 'rfft', 'fftfreq', 'rfftfreq', 'ifft', 'irfft']

class FFT():
    def __call__(self, input, norm=1, **kwargs):
        return torch.fft.fft(input, **kwargs) / norm

class RFFT():
    def __call__(self, input, norm=1, **kwargs):
        ft = torch.fft.rfft(input, **kwargs) / norm
        ft[...,1:]*=2
        return ft
    
class FFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.fftfreq(input, **kwargs)
    
class RFFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.rfftfreq(input, **kwargs)
    
class IFFT():
    def __call__(self, input, norm=1, **kwargs):
        return torch.fft.ifft(input, **kwargs) * norm
    
class IRFFT():
    def __call__(self, input, norm=1, **kwargs):
        ift = torch.fft.irfft(input, **kwargs) * norm
        ift[...,1:] /= 2
        return ift
    

fft   = FFT()
rfft  = RFFT()
ifft  = IFFT()
irfft = IRFFT()
fftfreq  = FFTFREQ()
rfftfreq = RFFTFREQ()
