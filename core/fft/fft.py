"""FFT Wrapper code to torch.fft.fft to reproduce same results as pycbc"""

import torch

class FFT():
    "Wrapper to torch.fft.fft"
    def __call__(self, input, fs = 1, **kwargs):
        return torch.fft.fft(input, **kwargs)/fs

class RFFT():
    "Wrapper to torch.fft.fft"
    def __call__(self, input, fs = 1, **kwargs):
        return torch.fft.rfft(input, **kwargs)/fs
    
class FFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.fftfreq(input, **kwargs)
    
class RFFTFREQ():
    def __call__(self, input, **kwargs):
        return torch.fft.rfftfreq(input, **kwargs)
    
class IFFT():
    def __call__(self, input, fs = 1, **kwargs):
        return torch.fft.ifft(input, **kwargs)*fs
    
class IRFFT():
    def __call__(self, input, fs = 1, **kwargs):
        return torch.fft.irfft(input, **kwargs)*fs
    

fft   = FFT()
rfft  = RFFT()
ifft  = IFFT()
irfft = IRFFT()
fftfreq  = FFTFREQ()
rfftfreq = RFFTFREQ()
    

    
if __name__=='__main__':
    
    x = torch.randn((1, 1000))
    f = rfft(x)
   #print(f)



