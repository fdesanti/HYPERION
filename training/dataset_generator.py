import torch
from torch.utils.data import Dataset

from ..core.fft import *



class DatasetGenerator(Dataset):
    """
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    
    Given a specified prior and a waveform generator it generates as output a tuple
    (parameters, whitened_strain)

    """

    def __init__(self, 
                 waveform_generator, 
                 prior, 
                 duration       = 1, 
                 noise_duration = 8, 
                 batch_size     = 512
                 ):
        """
        Constructor.

        Args:
        -----
        waveform_generator: object
            GWskysim Waveform generator object istance. (Example: the EffectiveFlyByTemplate generator)

        prior: object 
            hyperion's MultivariatePrior object instance defining the prior on the population parameters

        duration: float
            Duration (seconds) of the output strain. (Default: 1)

        noise_duration : float
            Duration (seconds) of noise to simulate. Setting it higher than duration helps to avoid
            border discontinuity issues when performing whitening. (Default: 8)
        
        batch_size : int
            Batch size dimension. (Default: 512)
        
        """
        super(DatasetGenerator, self).__init__()

        self.waveform_generator  = waveform_generator
        self.prior               = prior
        self.batch_size          = batch_size
        self.duration            = duration
        self.noise_duration      = noise_duration
        return

    @property
    def fs(self):
        return self.waveform_generator.fs
    
    @property
    def det_names(self):
        return self.waveform_generator.det_names


    def __len__(self):
        return int(1e8) #set it very high

    

    def __getitem__(self, idx=None):

        #sampling prior
        prior_samples = self.prior.sample((self.batch_size, 1))

        prior_samples['t0_p'] = prior_samples['t0_p'][0]
        h = self.waveform_generator(**prior_samples)

        n = self.noise_duration//2
        for ifo in self.det_names:
            dt = h[f"{ifo}_delay"]
            h_tmp  = torch.nn.functional.pad(h[ifo], (n*self.fs, n*self.fs), mode='replicate')  
            f = rfftfreq(h_tmp.shape[-1], d=1/self.fs)
            h_tmp = irfft(rfft(h_tmp, n = h_tmp.shape[-1], fs=self.fs) * torch.exp(-1j * 2 * torch.pi * f * dt), fs = self.fs)

            middle = h_tmp.shape[-1]//2

            
            h[ifo] = h_tmp[:, middle-self.fs//2:middle+self.fs//2]

            
        return h
    

