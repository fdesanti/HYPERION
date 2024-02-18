"""Random ASD generator. Implementation adapted from gwskysim gwcolorednoise"""

import torch
import numpy as np

from ..core.fft import rfftfreq
from ..config import CONF_DIR



class ASD_sampler():
    """
    Class that samples a new ASD from a reference one by generating random amplitude and phase for each frequency bin.

    Args:
    -----
        ifo : str
            Detector identifier (e.g. L1, H1 or V1). If given it will use the default O3 asd stored in the config dir.
            Mutually exclusive with asd_file

        asd_file : str 
            Path to asd (.txt) file specifying the reference asd.
        
        fs : float
            Sampling frequency of output generated ASD. (Default: 2048 Hz)

        duration : float
            Duration of strain timeseries (seconds). Used to compute the proper frequency array. (Default: 2)

        device: str
            Device on which perform the computation. Either 'cpu' or 'cuda:n' (Default: 'cpu')

        random_seed : int
            Random seed to set the random number generator for reproducibility. (Default: 123)
    
    Methods:
    --------
        - sample:
            Args:
            -----
                batch_size : int
                    Batch size for output sampled ASDs

            Returns:
            --------
                sampled_asd : torch.tensor
                    sampled ASD of shape [batch_size, fs//2+1]
        
        - __call__:
            wrapper to sample method

    """

    def __init__(self, ifo, asd_file=None, fs = 2048, duration=2, device = 'cpu', random_seed = 123):

        #read reference Asd
        if asd_file is not None:
            file = asd_file
        else:
            file = f"{CONF_DIR}/ASD_{ifo}_O3.txt"

        asd_f, asd = np.loadtxt(file, unpack=True)

        #generate frequency array
        self.f = rfftfreq(fs*duration, d=1/fs, device = device)
        self.df = torch.abs(self.f[1] - self.f[2])

        #reference ASD from interpolation
        self.asd_reference = torch.from_numpy(np.interp(self.f.cpu().numpy(), asd_f, asd)).to(device)

        #other attributes
        self.fs = fs
        self.device = device

        #set up random number generator
        self.rng = torch.Generator(device)
        self.rng.manual_seed(random_seed)
        return

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device_name):
        self._device = device_name

    @property
    def fs(self):
        return self._fs
    @fs.setter
    def fs(self, value):
        self._fs = value

    @property
    def asd_std(self):
        if not hasattr(self, '_asd_std'):
            self._asd_std = (0.5)* self.asd_reference * self.df ** 0.5
            #self._asd_std = self.asd_reference 
        return self._asd_std


    def sample(self, batch_size):

        out_asd_shape = (batch_size, len(self.f))
        
        # Generate scaled random power + phase
 
        mean = torch.zeros(out_asd_shape)
        out_asd_real = torch.normal(mean=mean, std=self.asd_std, generator=self.rng)
        out_asd_imag = torch.normal(mean=mean, std=self.asd_std, generator=self.rng)
    
        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
            #if not (samples % 2): si[..., -1] = 0

        # Regardless of signal length, the DC component must be real
        out_asd_imag[..., 0] = 0

        # Combine power + corrected phase to Fourier components
        out_asd = torch.abs(out_asd_real + 1J * out_asd_imag)

        #out_asd = torch.stack([self.asd_reference for _ in range(batch_size)])
        #out_asd = torch.mean(out_asd, axis = 0)

        return out_asd

    def __call__(self, batch_size = 512):
        return self.sample(batch_size)
    
    def _test(self):
        return 
