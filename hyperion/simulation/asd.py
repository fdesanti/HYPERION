"""Random ASD generator. Implementation adapted from gwskysim gwcolorednoise"""

import torch
import numpy as np

from ..config import CONF_DIR
from ..core.fft import rfftfreq, irfft

from scipy.interpolate import interp1d

class ASD_Sampler():
    """
    Class that samples a new ASD from a reference one by generating random amplitude and phase for each frequency bin.

    Args:
        ifo           (str): Detector identifier (e.g. L1, H1 or V1). If given it will use the default O3 asd stored in the config dir. Mutually exclusive with ``asd_file``.
        reference_run (str): Reference run of the detector. Mutually exclusive with ``asd_file``. (Default: 'O3')
        asd_file      (str): Path to asd (.txt) file specifying the reference asd. Mutually exclusive with ``ifo`` and ``reference_run``. (Default: None)
        fs          (float): Sampling frequency of output generated ASD. (Default: 2048 Hz)
        duration    (float): Duration of strain timeseries (seconds). Used to compute the proper frequency array. (Default: 2)
        device        (str): Device on which perform the computation. Either 'cpu' or 'cuda' (Default: 'cpu')
        random_seed   (int): Random seed to set the random number generator for reproducibility. (Default: None)
        fmin        (float): Minimum frequency to consider in the generated ASD. (Default: None)
    """

    def __init__(self, ifo=None, reference_run='O3', asd_file=None, fs=2048, duration=2, device = 'cpu', random_seed=None, fmin=None):

        #read reference Asd
        if asd_file is not None:
            file = asd_file
        else:
            file = f"{CONF_DIR}/ASD_curves/{reference_run}/ASD_{ifo}_{reference_run}.txt"

        asd_f, asd = np.loadtxt(file, unpack=True)

        #generate frequency array
        self.fs = fs
        self.noise_points = duration*fs

        self.f = rfftfreq(fs*duration, d=1/fs, device = device)
        
        f = self.f.clone()
        if fmin is not None:
            f[f < fmin] = fmin

        self.df = torch.abs(self.f[1] - self.f[2])
        
        #reference ASD from interpolation
        #self.reference_asd = torch.from_numpy(np.interp(self.f.cpu().numpy(), asd_f, asd)).to(device)
        asd_interp = interp1d(asd_f, asd, kind='cubic', fill_value='extrapolate')
        self.reference_asd = torch.from_numpy(asd_interp(f.cpu().numpy())).to(device)
        
        #other attributes
        self.device = device

        #set up random number generator
        self.rng = torch.Generator(device)
        if not random_seed:
            random_seed = torch.randint(0, 2**32, (1,)).item()
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
            self._asd_std = self.reference_asd * self.df ** 0.5
        return self._asd_std


    def sample(self, batch_size, noise=False, use_reference_asd=False):
        """
        Sample a new ASD from the reference one by generating random amplitude and phase for each frequency bin.

        Args:
            batch_size         (int): Batch size for output sampled ASDs
            noise             (bool): If True, return the noise timeseries generated from the sampled ASD. (Default: False)
            use_reference_asd (bool): If True, return the reference ASD. (Default: False)

        Returns:
            tuple: A tuple containing:
                - **sampled_asd** (torch.Tensor): Sampled ASD of shape [batch_size, fs//2+1].
                - **noise_from_asd** (torch.Tensor): Noise timeseries generated from the sampled ASD. (Only if ``noise``=True)
        """

        asd_shape = (batch_size, len(self.f))

        if use_reference_asd:
            asd = self.reference_asd * torch.ones((batch_size, 1), device = self.device)
                
            out_asd = (asd + 1j*asd) / np.sqrt(2)

        else:
            # Generate scaled random power + phase
            mean = torch.zeros(asd_shape, device = self.device, dtype=torch.float64)
            asd_real = torch.normal(mean=mean, std=self.asd_std, generator=self.rng)
            asd_imag = torch.normal(mean=mean, std=self.asd_std, generator=self.rng)
            
            # If the signal length is even, frequencies +/- 0.5 are equal
            # so the coefficient must be real.
            if not (self.noise_points % 2): asd_imag[..., -1] = 0

            # Regardless of signal length, the DC component must be real
            asd_imag[..., 0] = 0

            # Combine power + corrected phase to Fourier components
            out_asd = asd_real + 1J * asd_imag 
        
        #out_asd = torch.stack([self.reference_asd for _ in range(batch_size)])
        #out_asd = torch.mean(out_asd, axis = 0)
        if noise:
            noise_from_asd = irfft(out_asd, n=self.noise_points)
            #import matplotlib.pyplot as plt
            #plt.loglog(self.f.cpu(), abs(out_asd[0].cpu().numpy()))
            #plt.plot(noise_from_asd[0].cpu().numpy())
            #plt.show()
            return torch.abs(out_asd), noise_from_asd
        
        return torch.abs(out_asd)

    def __call__(self, batch_size = 1, noise=False, use_reference_asd=False):
        """
        Wrapper for the sample method.
        """
        return self.sample(batch_size, noise, use_reference_asd)
     
