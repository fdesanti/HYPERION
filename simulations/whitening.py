"""
Pytorch implementation of the Whitening.
Some of the functions are adapted from the GWPy library.
(See https://github.com/gwpy/gwpy/tree/main/gwpy)
"""

import torch
from torchaudio.functional import fftconvolve
from hyperion.core.fft import rfft, irfft, rfftfreq
from hyperion.core.fft.windows import tukey, planck, get_window

# ====================================================
# Helper functions ===================================
# ====================================================

def fir_from_transfer(transfer, ntaps, window='hann', ncorner=None):
    """Design a Type II FIR filter given an arbitrary transfer function

    Args:
    -----
        transfer (tensor)    : transfer function to start from, must have at least ten samples
        ntaps (int)          : number of taps in the final filter, must be an even number
        window (str, tensor) : (optional) window function to truncate with. (Default: 'hann')
        ncorner (int)        : (optional) number of extra samples to zero off at low frequency. (Default: 'None')
        
    Returns:
    --------
        out (tensor) : A time domain FIR filter of length 'ntaps'

    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = torch.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out

def truncate_transfer(transfer, ncorner=None):
    """Smoothly zero the edges of a frequency domain transfer function

    Args:
    -----
        transfer (tensor) :  transfer function to start from, must have at least ten samples
        ncorner (int)     :  (optional) number of extra samples to zero off at low frequency. (Default: 'None')
        
    Returns:
    --------
        out (tensor) :  the smoothly truncated transfer function
    """
    #recall that transfer has shape [batch_size, nsamp]
    nsamp = transfer.size()[-1]
    ncorner = ncorner if ncorner else 0
    out = transfer.clone()
    out[:, 0:ncorner] = 0
    out[:, ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5, device=transfer.device)
    return out

def truncate_impulse(impulse, ntaps, window='hann'):
    """Smoothly truncate a time domain impulse response

    Args:
    -----
    impulse (tensor):  the impulse response to start from
    ntaps   (int)   :  number of taps in the final filter
    window  (str)   : (optional) window function to truncate with. (Default: 'hann')
        
    Returns:
    --------
        out (tensor): the smoothly truncated impulse response
    """
    
    out  = impulse.clone()
    size = out.size()[-1]
    trunc_start = int(ntaps / 2)
    trunc_stop  = size - trunc_start
   
    window = get_window(window, window_length=ntaps)
    
    out[:, 0:trunc_start]   *= window[trunc_start:ntaps]
    out[:, trunc_stop:size] *= window[0:trunc_start]
    out[:, trunc_start:trunc_stop] = 0
    return out


def convolve(signal, fir, window='hann'):
    """
    Convolves a time domain timeseries with a FIR filter.
    
    Args:
    -----
        signal (tensor) : input time series
        fir    (tensor) : FIR filter
        window (str)    : window function to apply to the FIR filter. (Default is 'hann')

    Returns:
    --------
        conv (tensor)    : convolved time series
        
    Note:
    -----
        Instead of scipy.signal.fftconvolve, we use torchaudio.functional.fftconvolve.
    """
    #get sizes
    fir_size = fir.size()[-1]
    signal_size = signal.size()[-1]
    
    pad  = int(fir_size/2)
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
#============== WhitenNet Class  =====================
#=====================================================                              
class WhitenNet:
    """
    Class that performs whitening and adds Gaussian Noise to the data.
    It can exploit GPU acceleration.

    Constructor Args:
    -----------------
        fs (float)           : sampling frequency.
        duration (float)     : duration of the signal.
        rng (torch.Generator): random number generator. (Default is None).
        device (str)         : device to use for computations: either 'cpu' or 'cuda'. 
                               (Default is 'cpu').
    """

    def __init__(self, fs, duration, rng=None, device='cpu'):

        self.fs = fs
        self.device = device
        self.duration = duration
    
        self.rng = rng

        return

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

    
    def add_gaussian_noise(self, h):
        """
        Adds gaussian noise to the whitened signal(s).
        To ensure that noise follows a N(0, 1) distribution we divide by the noise standard deviation
        given by 1/sqrt(2*delta_t) where delta_t is the sampling interval.
        
        Args:
        -----
            h (dict of torch.Tensor): whitened signals
        """
        
        for det in h.keys():
            noise = torch.normal(mean=self.noise_mean, 
                                 std =self.noise_std, 
                                 generator=self.rng)
            h[det] += noise#/self.noise_std
            
        return h
    
    
    def whiten(self, h, asd, time_shift, noise=None, add_noise=True, 
               fduration=2, window='hann', ncorner=0):
        """
        Whiten the input signal and (optionally) add Gaussian noise.
        Whitening is performed by dividing the signal by its ASD in the frequency domain.

        Args:
        -----
            h (TensorDict)          : input signal(s) from different detectors
            asd (TensorDict)        : ASD of the noise for each detector
            time_shift (TensorDict) : time shift for each detector
            noise (TensorDict)      : Gaussian noise to add to the input template(s) - Mutually 
                                      exclusive with the 'add_noise' argument.
            add_noise (bool)        : whether to add Gaussian noise to the whitened signal(s)

        Returns:
        --------
            whitened (TensorDict)   : whitened signal(s) (with added noise).
        """
    
        #hf = {}
        whitened = h.copy()

        for det in h.keys():
            ht = h[det] * tukey(self.n, alpha=0.01, device=self.device)
            
            if noise:
                ht += noise[det]

            #compute the frequency domain signal (template) and apply time shift
            hf = rfft(ht, n=self.n, fs=self.fs) * torch.exp(-2j * torch.pi * self.freqs * time_shift[det]) 

            ht = irfft(hf, n=self.n, fs=self.fs)
            
            #if noise:
            #    ht += noise[det]

            #whiten the signal by dividing wrt the ASD
            #hf_w = hf / asd[det]
            ntaps = int((fduration * self.fs))
            fir   = fir_from_transfer(1/asd[det], ntaps=ntaps, window=window, ncorner=ncorner)
            whitened[det] = convolve(ht, fir, window=window) #/ self.noise_std

            #convert back to the time domain
            # we divide by the noise standard deviation to ensure to have unit variance
            #whitened[det] = irfft(hf_w, n=self.n, fs=self.fs) / self.noise_std
        
        #compute the optimal SNR
        #snr = network_optimal_snr(hf, self.PSDs, self.duration) / self.fs
        if add_noise and not noise:
            whitened = self.add_gaussian_noise(whitened)
        
        return whitened
    

    def __call__(self, **whiten_kwargs):
        return self.whiten(**whiten_kwargs)
        

