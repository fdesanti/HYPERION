"""Implementation of a Standard Gravitational Wave Gaussian Likelihood. (See arxiv.org/pdf/1809.02293 (eq. 44)"""
""""""
# We implement the Gaussian likelihood as follows:
#
#       logL(d|theta) = sum_k logL(d_k|theta) = 
#                      = - 1/2 sum_k log(2piPSD_k) - 2*df sum_k |d_k - h_k(theta)|^2 / PSD_k
#                      = psi - 1/2 <d - h(theta)|d - h(theta)>

# where psi = - 1/2 sum_k log(2piPSD_k) can be discarded being a constant

# By expanding the inner product we get:
#
#       logL(d|theta) = log Zn + kappa^2(theta) - 1/2 rho_opt^2(theta)
#
# where:
#               -2log Zn = <d|d>               is the noise log likelihood (Zn is the noise evidence)
#
#       rho_opt^2(theta) = <h(theta)|h(theta)> is the optimal signal-to-noise ratio
#
#         kappa^2(theta) = <d|h(theta)>        is the signal-to-noise ratio (matched filter)


import torch
import multiprocess as mp
mp.set_start_method('spawn', force=True) # It only works with 'spawn' method when doing inference

from ...core.fft import rfft, rfftfreq

pi = torch.tensor(torch.pi)

class GWLikelihood():
    """
    Gravitational Wave Likelihood class
    
    Args:
    -----
        waveform_generator : Waveform generator object which returns ifo injected strain
        device (str)       : Device to run the likelihood computation. (Default: 'cpu')
    """
    
    def __init__(self, waveform_generator, device = 'cpu'):

        self.waveform_generator=waveform_generator
        self.device = device
        
        return
    
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
    
    
    def _inner_product(self, a, b, N=1, psd=None):
        """
        Computes the inner product between two frequency series. 
        Works with batched data.

        The (noise weighted) inner product between two frequency series a and b is defined as:
        
               <a|b> = 4 df sum_k Re(a.cong()_k * b_k / PSD_k)
        
        Args:
        -----
            a (torch.Tensor): Frequency series
            b (torch.Tensor): Frequency series
            psd (torch.Tensor): Power Spectral Density. If None, the inner product is not weighted by the PSD

        Returns:
        --------
            inner_product (torch.Tensor): Inner product between a and b
        
        Note:
        -----
            If PSD is None, the inner product is not weighted by the PSD: i.e. we assume that a and b are
                                                                          already whitened
        
        """
        integrand = (a.conj() * b / psd)
        '''
        if psd is not None: 
            psd = psd.type(a.type()) #recast to complex dtype if necessary
            integrand /= psd
        '''

        return (4 / self.duration) * torch.sum(integrand.real) 
    

    def noise_log_Likelihood(self, strain, psd):
        """
        Computes the log Likelihood assuming strain contains only Gaussian noise
        
        
        Args:
        -----
            strain (dict) : Dictionary containing interferometer strain time series
            psd (dict)    : Dictionary containing interferometer Power Spectral Densities
                
        Returns:
        --------
            logZn (float) : Noise Log Likelihood 
        """
        
        logZn = 0.0
        #import matplotlib.pyplot as plt
        #plt.figure()
        for ifo in strain.keys():
            N = strain[ifo].shape[-1]
            print("fs: ", self.fs)
            mask = (self.frequencies > 10) * (self.frequencies < 1000)
            frequency_domain_strain =  rfft(strain[ifo], n=N, norm=self.fs) / self.duration
            
            logZn -= 0.5 * self._inner_product(frequency_domain_strain[mask], frequency_domain_strain[mask], N, psd[ifo][mask])
            #plt.loglog(self.frequencies[mask].cpu().numpy(), torch.abs(frequency_domain_strain[mask]).cpu().numpy(), label=ifo)
        #plt.legend()
        #plt.savefig("psd.png", bbox_inches='tight')
        
        print(f"Noise Log Likelihood: {logZn.real}")
        return logZn.real
    

    def log_Likelihood(self, strain, theta, psd=None):
        """
        Computes the log Likelihood assuming strain contains a GW signal
        
        Args:
        -----            
            strain (dict)  : Dictionary containing interferometer strain time series
            theta (dict)   : Dictionary containing the GW parameters
            psd (dict)     : Dictionary containing interferometer Power Spectral Densities
                
        Returns:
        --------
            logL (float)   : Log Likelihood 
        """

        #compute the noise log likelihood
        logZn = self.noise_log_Likelihood(strain, psd)

        #compute the frequency domain template waveforms
        frequency_domain_template = self.get_frequency_domain_template(theta)
        
        #compute the log likelihood
        logL = 0.0
        for ifo in strain.keys():
            N = strain[ifo].shape[-1]
            
            frequency_domain_strain = rfft(strain[ifo], norm=self.fs)

            kappa    = self._inner_product(frequency_domain_strain, frequency_domain_template[ifo], N, psd[ifo])
            rho_opt  = self._inner_product(frequency_domain_template[ifo], frequency_domain_template[ifo], N, psd[ifo])
            logL_ifo = kappa - 0.5 * rho_opt

            logL    += logL_ifo

        print(f"Log Likelihood: {logL.real + logZn}")

        return logL.real + logZn
    
    
    
    
    
    def get_frequency_domain_template(self, theta):
        """
        Computes the frequency domained templates using the waveform_generator object
        
        Args:
        -----
            theta (dict): Dictionary with each key representing a gw parameter
                
        Returns:
        --------
            frequency_domain_template (TensorDict) : Dictionary (with ifos as keys) containing the projected frequency domain template 
        
        """
        template, tcoal = self.waveform_generator(theta, project_onto_detectors=True)
        
        time_delays = self.det_network.time_delay_from_earth_center(theta['ra'], 
                                                                    theta['dec'])
        tcoal  = theta['tcoal'].to(self.device) - tcoal.squeeze(1).to(self.device)
        
        
        n = self.waveform_generator.duration * self.fs
        frequency_domain_template = dict()
        for ifo in self.det_network.names:

            hf  =  rfft(template[ifo], norm=self.fs)
            
            dt = (tcoal + time_delays[ifo]).unsqueeze(-1)
                        
            #take into account the time shift
            frequency_domain_template[ifo] = hf * torch.exp(-1j * 2 * torch.pi * self.frequencies * dt) 

            frequency_domain_template[ifo] *= torch.exp(-1j*pi/2)
        
        return frequency_domain_template