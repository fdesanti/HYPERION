import torch
import multiprocess as mp
mp.set_start_method('spawn', force=True) # It only works with 'spawn' method when doing inference

from ..core.fft import rfft, rfftfreq

pi = torch.tensor(torch.pi)

class GWLikelihood:
    r"""
    Standard Gravitational Wave Gaussian Likelihood class implementation (see arxiv.org/pdf/1809.02293 (eq. 44) 
    We implement the Gaussian likelihood as follows:

    .. math::

        \log L(d|\theta) = \sum_k \log L(d_k|\theta)
                        = -\frac{1}{2} \sum_k \log(2\pi\,\mathrm{PSD}_k)
                            - 2\,\Delta f \sum_k \frac{\left| d_k - h_k(\theta) \right|^2}{\mathrm{PSD}_k}
                        = \psi - \frac{1}{2}\langle d - h(\theta) \,|\, d - h(\theta) \rangle

    where

    .. math::

        \psi = -\frac{1}{2} \sum_k \log(2\pi\,\mathrm{PSD}_k)

    can be discarded as a constant.

    By expanding the inner product we get:

    .. math::

        \log L(d|\theta) = \log Z_n + \kappa^2(\theta) - \frac{1}{2}\rho_{\rm opt}^2(\theta)

    where:

    .. math::

        -2\log Z_n = \langle d|d \rangle 
        \qquad \text{(noise log likelihood, with $Z_n$ being the noise evidence)}

    .. math::

        \rho_{\rm opt}^2(\theta) = \langle h(\theta)|h(\theta) \rangle 
        \qquad \text{(optimal signal-to-noise ratio)}

    and

    .. math::

        \kappa^2(\theta) = \langle d|h(\theta) \rangle 
        \qquad \text{(signal-to-noise ratio, matched filter)}
    
    Args:
        waveform_generator  : Hyperion's Waveform generator object which returns ifo injected strain
        device        (str) : Device to run the likelihood computation. (Default: 'cpu')
    """
    
    def __init__(self, waveform_generator, device='cpu', fmin=10, fmax=1000):

        self.waveform_generator=waveform_generator
        self.device = device
        self.frequency_mask = (self.frequencies >= fmin) * (self.frequencies <= fmax)
        
    
    @property
    def det_network(self):
        if not hasattr(self.waveform_generator, 'det_network'):
            raise AttributeError('Waveform generator must have a detector network')
        return self.waveform_generator.det_network
    
    @property
    def fs(self):
        if self.waveform_generator:
            return self.waveform_generator.fs
        elif hasattr(self, '_fs'):
            return self._fs
        else:
            raise AttributeError('Sampling frequency must be set')
    @fs.setter
    def fs(self, value):
        self._fs = value
        
    @property
    def duration(self):
        if self.waveform_generator:
            return self.waveform_generator.duration 
        elif hasattr(self, '_duration'):
            return self._duration 
        else:
            raise AttributeError('Duration must be set')
    
    @duration.setter
    def duration(self, value):
        self._duration = value
    
    @property
    def frequencies(self):
        return rfftfreq(self.duration * self.fs, d=1/self.fs, device=self.device)
    
    @property
    def delta_f(self):
        return self.frequencies[1] - self.frequencies[0]
    
    
    def _inner_product(self, a, b, psd=None):
        """
        Computes the inner product between two frequency series. 
        Works with batched data.

        The (noise weighted) inner product between two frequency series a and b is defined as:

        .. math::
            \langle a | b \rangle = 4 \, \Delta f \, \sum_{f} \frac{a^*(f) \, b(f)}{\mathcal{S}_n(f)}
        
        Args:
            a   (torch.Tensor): Frequency series
            b   (torch.Tensor): Frequency series
            psd (torch.Tensor): Power Spectral Density. If None, the inner product is not weighted by the PSD

        Returns:
            inner_product (torch.Tensor): Inner product between a and b
        
        Note:
            If PSD is None, the inner product is not weighted by the PSD: i.e. we assume that a and b are already whitened
        """

        integrand = a.conj() * b
        
        if psd is not None: 
            psd = psd.type(a.type()) #recast to complex dtype if necessary
            integrand /= psd

        return (4 / self.duration) * torch.sum(integrand.real) 
    
    def get_frequency_domain_template(self, theta):
        """
        Computes the frequency domained templates using the waveform_generator object
        
        Args:
            theta (dict): Dictionary with each key representing a gw parameter
                
        Returns:
            frequency_domain_template (TensorDict): Dictionary (with ifos as keys) containing the projected frequency domain template 
        
        """
        #generate the projected waveform
        template, tcoal = self.waveform_generator(theta, project_onto_detectors=True)
        
        #compute time delays wrt to the earth center & adjust the time of coalescence
        time_delays = self.det_network.time_delay_from_earth_center(theta['ra'], 
                                                                    theta['dec'])
        tcoal  = theta['tcoal'].to(self.device) - tcoal.squeeze(1).to(self.device)
        
        #compute the fourier transform of the waveform
        n = self.waveform_generator.duration * self.fs
        frequency_domain_template = dict()
        for ifo in self.det_network.names:
            N = template[ifo].shape[-1]
            
            hf  =  rfft(template[ifo], n=N, norm=self.fs)
            
            #take into account the time shift
            dt = (tcoal + time_delays[ifo]).unsqueeze(-1)

            frequency_domain_template[ifo] = hf * torch.exp(-1j * 2 * pi * self.frequencies * dt) 
        
        return frequency_domain_template
    

    def noise_log_Likelihood(self, strain, psd):
        """
        Computes the log Likelihood assuming strain contains only Gaussian noise
        
        Args:
            strain (dict): Dictionary containing interferometer strain time series
            psd    (dict): Dictionary containing interferometer Power Spectral Densities
                
        Returns:
            logZn (float): Noise Log Likelihood 
        """
        
        logZn = 0.0
        #import matplotlib.pyplot as plt
        #plt.figure()
        for ifo in strain.keys():
            N = strain[ifo].shape[-1]
            
            frequency_domain_strain =  rfft(strain[ifo], n=N, norm=self.fs) / self.duration
            
            logZn -= 0.5 * self._inner_product(frequency_domain_strain[self.frequency_mask], 
                                               frequency_domain_strain[self.frequency_mask], 
                                               psd[ifo][self.frequency_mask])
            #plt.loglog(self.frequencies[mask].cpu().numpy(), torch.abs(frequency_domain_strain[mask]).cpu().numpy(), label=ifo)
        #plt.legend()
        #plt.savefig("psd.png", bbox_inches='tight')
        
        print(f"Noise Log Likelihood: {logZn.real}")
        return logZn.real
    

    def log_Likelihood(self, strain, theta, psd=None):
        """
        Computes the log Likelihood assuming strain contains a GW signal. 
        (see Eq. 44 of arxiv.org/pdf/1809.02293)
        
        Args:         
            strain (dict): Dictionary containing interferometer strain time series
            theta  (dict): Dictionary containing the GW parameters
            psd    (dict): Dictionary containing interferometer Power Spectral Densities
                
        Returns:
            logL (float): Log Likelihood 
        """

        #compute the noise log likelihood
        logZn = self.noise_log_Likelihood(strain, psd)

        #compute the frequency domain template waveforms
        frequency_domain_template = self.get_frequency_domain_template(theta)
        
        #compute the log likelihood
        logL = 0.0
        for ifo in strain.keys():
            N = strain[ifo].shape[-1]
            
            frequency_domain_strain = rfft(strain[ifo], n=N, norm=self.fs) / self.duration

            kappa    = self._inner_product(frequency_domain_strain, frequency_domain_template[ifo], psd[ifo])
            rho_opt  = self._inner_product(frequency_domain_template[ifo], frequency_domain_template[ifo], psd[ifo])
            logL_ifo = kappa - 0.5 * rho_opt

            logL += logL_ifo

        print(f"Log Likelihood: {logL.real + logZn}")

        return logL.real + logZn