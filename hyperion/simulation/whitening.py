"""
Pytorch implementation of the Whitening.
Some of the functions are adapted from the GWPy library.
(See https://github.com/gwpy/gwpy/tree/main/gwpy)
"""

import torch
import numpy as np
from torchaudio.functional import fftconvolve
from hyperion.core.fft import rfft, irfft, rfftfreq
from hyperion.core.fft.windows import tukey, planck, get_window

# ====================================================
# Helper functions ===================================
# ====================================================

def fir_from_transfer(transfer, ntaps, window='hann', ncorner=None):
    """Design a Type II FIR filter given an arbitrary transfer function

    Args:
        transfer     (torch.Tensor): transfer function to start from, must have at least ten samples
        ntaps                 (int): number of taps in the final filter, must be an even number
        window  (str, torch.Tensor): (Optional) window function to truncate with. (Default: 'hann')
        ncorner               (int): (Optional) number of extra samples to zero off at low frequency. (Default: 'None')
        
    Returns:
        out (torch.Tensor): A time domain FIR filter of length 'ntaps'

    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = torch.fft.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = torch.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out

def truncate_transfer(transfer, ncorner=None):
    """Smoothly zero the edges of a frequency domain transfer function

    Args:
        transfer (torch.Tensor): Transfer function to start from, must have at least ten samples
        ncorner           (int): (Optional) number of extra samples to zero off at low frequency. (Default: 'None')
        
    Returns:
        out (torch.Tensor): the smoothly truncated transfer function
    """
    #recall that transfer has shape [batch_size, nsamp]
    nsamp = transfer.size()[-1]
    ncorner = ncorner if ncorner else 0
    out = transfer.clone()
    out[..., 0:ncorner] = 0
    out[..., ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5, device=transfer.device)
    return out

def truncate_impulse(impulse, ntaps, window='hann'):
    """Smoothly truncate a time domain impulse response

    Args:
        impulse (torch.Tensor): impulse response to start from
        ntaps           (int): number of taps in the final filter
        window          (str): window function to apply to the FIR filter. (Default is 'hann')

    Returns:
        out (torch.Tensor): the smoothly truncated impulse response
    """
    
    out  = impulse.clone()
    size = out.size()[-1]
    trunc_start = int(ntaps / 2)
    trunc_stop  = size - trunc_start
   
    window = get_window(window, window_length=ntaps)
    
    out[..., 0:trunc_start]   *= window[trunc_start:ntaps]
    out[..., trunc_stop:size] *= window[0:trunc_start]
    out[..., trunc_start:trunc_stop] = 0
    return out


def convolve(signal, fir, window='hann'):
    """
    Convolves a time domain timeseries with a FIR filter.
    
    Args:
        signal (torch.Tensor): input signal to convolve
        fir    (torch.Tensor): FIR filter to convolve with
        window          (str): window function to apply to the FIR filter. (Default is 'hann')

    Returns:
        conv (torch.Tensor): the convolved signal
        
    Note:
        Instead of scipy.signal.fftconvolve, we use torchaudio.functional.fftconvolve.
    """
    #get sizes
    fir_size = fir.size()[-1]
    signal_size = signal.size()[-1]
    
    pad  = int(np.ceil(fir_size/2))
    nfft = min(8*fir_size, signal_size)
    
    # condition the input data
    in_ = signal.clone()
    w   = get_window(window, window_length=fir_size)
    in_[:,:pad]  *= w[:pad]
    in_[:,-pad:] *= w[-pad:]
    
    # if FFT length is long enough, perform only one convolution
    if nfft >= signal_size/2:
        conv = fftconvolve(in_, fir, mode='same')
    
    # else use the overlap-save algorithm
    else:
        nstep = nfft - 2*pad
        conv = torch.empty_like(in_)
        # handle first chunk separately
        conv[:,:nfft-pad] = fftconvolve(in_[:,:nfft], fir,
                                      mode='same')[:,:nfft-pad]
        # process chunks of length nstep
        k = nfft - pad
        while k < signal_size - nfft + pad:
            yk = fftconvolve(in_[:,k-pad:k+nstep+pad], fir,
                             mode='same')
            conv[:,k:k+yk.size-2*pad] = yk[:,pad:-pad]
            k += nstep
        # handle last chunk separately
        conv[:,-nfft+pad:] = fftconvolve(in_[:,-nfft:], fir,
                                       mode='same')[:,-nfft+pad:]
    return conv


#=====================================================
#==============  WhitenNet Class  ====================
#=====================================================                              
class WhitenNet:
    """
    Class that performs whitening and adds Gaussian Noise to the data.
    It can exploit GPU acceleration.

    Args:
        fs              (int): Sampling frequency [Hz]
        duration      (float): Duration of the data segment [s]
        device          (str): Device to use for computation (Default: 'cpu')
        rng (torch.Generator): Random number generator for adding noise
        
    """

    def __init__(self, fs, duration, device='cpu', rng=None):

        self.fs       = fs
        self.device   = device
        self.duration = duration
        self.rng      = rng

    @property
    def n(self):
        return self.duration * self.fs
       
    @property
    def delta_t(self):
        if not hasattr(self, '_delta_t'):
            self._delta_t = torch.as_tensor(1/self.fs).to(self.device)
        return self._delta_t
    
    @property
    def noise_mean(self):
        if not hasattr(self, '_mu'):
            self._mu = torch.zeros(self.n, device=self.device)
        return self._mu
    
    @property
    def noise_std(self):
        """Rescaling factor to have gaussian noise with unit variance."""
        if not hasattr(self, '_noise_std'):
            self._noise_std = 1 / torch.sqrt(2*self.delta_t)
        return self._noise_std
    
    @property
    def freqs(self):
        if not hasattr(self, '_freqs'):
            self._freqs = rfftfreq(self.n, d=1/self.fs, device=self.device)
        return self._freqs
    
    @property
    def delta_f(self):
        if not hasattr(self, '_delta_f'):
            self._delta_f = self.freqs[1]-self.freqs[0]
        return self._delta_f

    
    def add_gaussian_noise(self, h, normalize):
        """
        Adds gaussian noise to the whitened signal(s).
        To ensure that noise follows a N(0, 1) distribution we divide by the noise standard deviation
        given by 1/sqrt(2*delta_t) where delta_t is the sampling interval.
        
        Args:
            h (dict, TensorDict): whitened signals
            normalized     (bool): whether to normalize the noise to have unit variance
        """
        for det in h.keys():

            noise = torch.normal(mean=torch.ones_like(h[det])*self.noise_mean, 
                                 std =torch.ones_like(h[det])*self.noise_std, 
                                 generator=self.rng)
            
            if normalize:
                noise /= self.noise_std
            h[det] += noise 

        return h
    
    def whiten(self, h, asd, time_shift=None, add_noise=False, noise=None,
               fduration=None, window='hann', ncorner=0, normalize=True, method='gwpy'):
        """
        Whiten the input signal and (optionally) add Gaussian noise.
        Whitening is performed by dividing the signal by its ASD in the frequency domain.
        If method='gwpy', the whitening is performed using a FIR filter designed from the ASD as in the gwpy library. 
        Otherwise, the whitening is performed by dividing the signal by the ASD in the frequency domain.

        Args:
            h          (dict, TensorDict): Input signal(s) to whiten
            asd        (dict, TensorDict): ASD of the noise
            time_shift (dict, TensorDict): time shift for each detector to apply to the signal
            add_noise              (bool): Whether to add Gaussian noise to the whitened signal. If True, white noise is added. (Default: False)
            noise      (dict, TensorDict): Gaussian noise to add to the signal. Mutually exclusive with add_noise
            fduration             (float): Duration of the filter to use for the whitening (only for method='gwpy')
            window                  (str): Window function to use for the FIR filter (only for method='gwpy')
            ncorner                 (int): Number of extra samples to zero off at low frequency (only for method='gwpy')
            normalize              (bool): Whether to normalize the noise to have unit variance
            method                  (str): Method to use for whitening ('gwpy' or 'pycbc'). (Default: 'gwpy')
        """
        with torch.device(self.device):
            fft_norm = self.n if method == 'gwpy' else self.fs

            #define the output whitened strain tensordict
            #we copy h so that it lies on the same device as the input
            whitened = h.copy()
            
            for det in h.keys():
                ht = h[det] #* tukey(self.n, alpha=0.01, device=self.device)
                if noise:
                    ht += noise[det]
                
                #compute the frequency domain signal (template) and (optionally) apply time shift
                shift = time_shift[det] if time_shift is not None else 0
                hf = rfft(ht, n=self.n, norm=fft_norm) * torch.exp(-2j * torch.pi * self.freqs * shift) 

                #gwpy whitening method
                if method == 'gwpy':
                    if not fduration:
                        fduration = self.duration
                    nout  = (hf.size()[-1] - 1) * 2
                    ht    = irfft(hf*nout, n=nout)   #NOTE - for this ifft method see https://github.com/gwpy/gwpy/blob/main/gwpy/frequencyseries/frequencyseries.py
                    ntaps = int((fduration * self.fs))
                    fir   = fir_from_transfer(1/asd[det], ntaps=ntaps, window=window, ncorner=ncorner)

                    whitened[det] = convolve(ht, fir, window=window)
                
                else:
                    #whiten the signal by dividing wrt the ASD
                    hf_w = hf / asd[det]
                    
                    #convert back to the time domain
                    # we divide by the noise standard deviation to ensure to have unit variance
                    whitened[det] = irfft(hf_w, n=self.n, norm=fft_norm)
                
                if normalize:
                    whitened[det] /= self.noise_std
            
            if add_noise and not noise:
                whitened = self.add_gaussian_noise(whitened, normalize)
        
        return whitened
    
    def __call__(self, **whiten_kwargs):
        """Wrapper to the whiten method."""
        return self.whiten(**whiten_kwargs)