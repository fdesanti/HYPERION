"""Definition of useful functions for the simulations"""

import torch

from tqdm import tqdm
from ..core.fft import rfft

import astropy.units as u
import astropy.cosmology as cosmo
from astropy.cosmology import Planck18, z_at_value

#======================================
# Cosmology
#======================================

def luminosity_distance_from_redshift(z, cosmology=Planck18):
    """
    Computes the luminosity distance from the redshift assuming a given cosmology.
    
    Args:
        z         (float, torch.Tensor): Redshift
        cosmology   (astropy.cosmology): Cosmology object. (Default is Planck18).
    """
    return cosmology.luminosity_distance(z).value

def redshift_from_luminosity_distance(dl, cosmology=Planck18):
    """
    Computes the redshift from the luminosity distance assuming a given Cosmology.
    
    Args:
        dl        (float, torch.Tensor): Luminosity distance in Mpc
        cosmology   (astropy.cosmology): Cosmology object. (Default is Planck18).
    """
    assert isinstance(cosmology, cosmo.FlatLambdaCDM), "cosmology must be an astropy cosmology object"
    
    z = dl
    for i in tqdm(range(len(dl)), total=len(dl), ncols = 100, ascii=' ='):
        z[i] = z_at_value(cosmology.luminosity_distance, dl[i] * u.Mpc)
    return z

#======================================
# Matched filter  & SNR
#======================================

def noise_weighted_inner_product(a, b, psd, duration):
    r"""
    Computes the noise weighted inner product of two frequency domain signals a and b.

    .. math::

        \langle a|b \rangle =  \dfrac{4}{T} \int_{0}^{\infty} \dfrac{\tilde{a}^*(f) \tilde{b}(f)}{S_n(f)} df

    Args:
        a   (torch.Tensor): Frequency domain signal
        b   (torch.Tensor): Frequency domain signal
        psd (torch.Tensor): Power spectral density
        duration   (float): Duration of the signal
    """
    integrand = torch.conj(a) * b / psd
    return (4 / duration) * torch.sum(integrand, dim = -1)


def optimal_snr(frequency_domain_template, psd, duration):
    r"""
    Computes the optimal SNR of a signal.
    
    .. math::
    
        \rho_{opt} = \sqrt{\langle \tilde{h}|\tilde{h}\rangle}
    
    Args:
        frequency_domain_template (torch.Tensor): Frequency domain signal
        psd                       (torch.Tensor): Power spectral density
        duration                         (float): Duration of the signal
    """
    rho_opt = noise_weighted_inner_product(frequency_domain_template, 
                                           frequency_domain_template, 
                                           psd, duration)
    
    snr_square = torch.abs(rho_opt)
    return torch.sqrt(snr_square)
    
    
def matched_filter_snr(frequency_domain_template, frequency_domain_strain, psd, duration):
    r"""
    Computes the matched filter SNR of a signal.
    
    .. math::
    
        \rho^2 = \dfrac{\langle \tilde{h}|\tilde{s}\rangle}{\sqrt{\langle \tilde{h}|\tilde{h}\rangle}}

    Args:
        frequency_domain_template (torch.Tensor): Frequency domain template signal
        frequency_domain_strain   (torch.Tensor): Frequency domain signal
        psd                       (torch.Tensor): Power spectral density
        duration                         (float): Duration of the signal

    """
    rho = noise_weighted_inner_product(frequency_domain_template, 
                                       frequency_domain_strain, 
                                       psd, duration)
    
    rho_opt = noise_weighted_inner_product(frequency_domain_template, 
                                           frequency_domain_template, 
                                           psd, duration)
    
    snr_square = torch.abs(rho / torch.sqrt(rho_opt))
    return torch.sqrt(snr_square)


def network_optimal_snr(frequency_domain_strain, psd, duration):
    r"""
    Computes the network SNR of a signal given by
    
    .. math::

        \rho_{net} = \sqrt{\sum_{det} \rho_{det}^2}

    Args:
        frequency_domain_strain (dict, TensorDict): Frequency domain signals
        psd                         (torch.Tensor): Power spectral density
        duration                           (float): Duration of the signal
    """
    
    snr = 0
    for det in frequency_domain_strain.keys():
        snr += optimal_snr(frequency_domain_strain[det], psd[det], duration)**2
       
    return torch.sqrt(snr)
        

def rescale_to_network_snr(h, new_snr, old_snr=None, **kwargs):
    """
    Rescales the input signal to a new network SNR. 
    If not provided, the old network SNR will be computed.
    Then the rescaling is done by multiplying the signal by the ratio between the new and old SNR.
    
    Args:
        h          (dict, TensorDict): Time domain signals. 
        new_snr (float, torch.Tensor): New network SNR
        old_snr (float, torch.Tensor): Old network SNR. If None it will be computed.
        kwargs                       : Additional arguments to pass to the optimal_snr function (e.g. the sampling frequency to compute the fft)
    
    Returns:
        hnew (dict, TensorDict): rescaled time domain signals
    """

    h_o = h.copy()
    
    if old_snr is None:
        #get kwargs
        fs = kwargs.get('fs')
        psd = kwargs.get('psd')
        duration = kwargs.get('duration')
        
        #compute the fft of the signals
        hf = torch.stack([h[key] for key in h])      #we stack the various waveforms together
        hf = rfft(hf, n=hf.shape[-1], fs=fs)         #as pytorch is faster with batched ffts
        hf_dict = {key: hf[i] for i, key in enumerate(h)} #then we reconvert to a dictionary

        #compute the old snr
        old_snr = network_optimal_snr(hf_dict, psd, duration)

    #perform rescaling   
    for det in h:
        h_o[det] *= (new_snr/old_snr)
        
    return h_o