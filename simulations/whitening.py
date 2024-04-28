import torch
from hyperion.core.fft import (tukey,
                               rfft, 
                               irfft, 
                               rfftfreq)
                               
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
    def window(self):
        if not hasattr(self, '_window'):
            self._window = tukey(self.n, alpha=0.01, device=self.device)
        return self.window
    
       
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
            self._freqs = rfftfreq(self.n, d=1/self.fs)
        return self._freqs

    
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
                                 std=self.noise_std, 
                                 generator=self.rng)
            h[det] += noise/self.noise_std
            
        return h
    
    
    def whiten(self, h, asd, time_shift, add_noise=True):
        """
        Whiten the input signal and (optionally) add Gaussian noise.
        Whitening is performed by dividing the signal by its ASD in the frequency domain.

        Args:
        -----
            h (TensorDict)          : input signal(s) from different detectors
            asd (TensorDict)        : ASD of the noise for each detector
            time_shift (TensorDict) : time shift for each detector
            add_noise (bool)        : whether to add Gaussian noise to the whitened signal(s)

        Returns:
        --------
            whitened (TensorDict)   : whitened signal(s) (with added noise).
        """
    
        #hf = {}
        whitened = h.copy()

        for det in h.keys():
            ht = h[det] * self.window

            #compute the frequency domain signal (template) and apply time shift
            hf = rfft(ht, n=self.n, fs=self.fs) * torch.exp(-2j * torch.pi * self.freqs * time_shift[det]) 

            #whiten the signal by dividing wrt the ASD
            hf_w = hf / asd[det]

            #convert back to the time domain
            # we divide by the noise standard deviation to ensure to have unit variance
            whitened[det] = irfft(hf_w, n=self.n, fs=self.fs) / self.noise_std
        
        #compute the optimal SNR
        #snr = network_optimal_snr(hf, self.PSDs, self.duration) / self.fs
        if add_noise:
            whitened = self.add_gaussian_noise(whitened)
        
        return whitened
    

    def __call__(self, **whiten_kwargs):
        return self.whiten(**whiten_kwargs)
        