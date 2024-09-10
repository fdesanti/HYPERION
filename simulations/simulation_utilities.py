"""Definition of useful functions for the simulations"""

import torch

from tqdm import tqdm
from ..core.fft import rfft

import astropy.units as u
from astropy.cosmology import Planck18, z_at_value

#======================================
# Cosmology
#======================================

def luminosity_distance_from_redshift(z, cosmology=Planck18):
    """
    Computes the luminosity distance from the redshift assuming a given cosmology.
    
    Args:
    -----
        z (float or torch.Tensor): redshift
        cosmology (astropy.cosmology): cosmology object. (Default is Planck18).
    """
    return cosmology.luminosity_distance(z).value

def redshift_from_luminosity_distance(dl, cosmology=Planck18):
    """
    Computes the redshift from the luminosity distance assuming a given Cosmology.
    
    Args:
    -----
        dl (float or torch.Tensor): luminosity distance in Mpc
        cosmology (astropy.cosmology): cosmology object. (Default is Planck18).
    """
    z = dl
    for i in tqdm(range(len(dl)), total=len(dl), ncols = 100, ascii=' ='):
        z[i] = z_at_value(cosmology.luminosity_distance, dl[i] * u.Mpc)
    return z

#======================================
# Matched filter  & SNR
#======================================

def noise_weighted_inner_product(a, b, psd, duration):
    """
    Computes the noise weighte inner product of two frequency domain signals a and b.
    
    Args:
    -----
        a (torch.Tensor): frequency domain signal
        b (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    
    """
    integrand = torch.conj(a) * b / psd
    return (4 / duration) * torch.sum(integrand, dim = -1)


def optimal_snr(frequency_domain_template, psd, duration):
    """
    Computes the optimal SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:
    -----
        frequency_domain_template (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    """
    rho_opt = noise_weighted_inner_product(frequency_domain_template, 
                                           frequency_domain_template, 
                                           psd, duration)
    
    snr_square = torch.abs(rho_opt)
    return torch.sqrt(snr_square)
    
    
def matched_filter_snr(frequency_domain_template, frequency_domain_strain, psd, duration):
    """
    Computes the matched filter SNR of a signal.
    The code is adapted from Bilby 
    (https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/gw/utils.py?ref_type=heads)
    
    Args:
    -----
        frequency_domain_template (torch.Tensor): frequency domain template signal
        frequency_domain_strain (torch.Tensor): frequency domain signal
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
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
    """
    Computes the network SNR of a signal given by
    
    SNR_net = [sum (snr_i ^2) ] ^(1/2)

    Args:
    -----
        frequency_domain_strain (dict of torch.Tensor): frequency domain signals
        psd (torch.Tensor): power spectral density
        duration (float): duration of the signal
    """
    
    snr = 0
    for det in frequency_domain_strain.keys():
        snr += optimal_snr(frequency_domain_strain[det], psd[det], duration)**2
       
    return torch.sqrt(snr)
        
    #return torch.sqrt(torch.sum(torch.stack(det_snr)**2, dim = -1))


def rescale_to_network_snr(h, new_snr, old_snr = None, **kwargs):
    """
    Rescales the input signal to a new network SNR. 
    
    
    Args:
    -----
        h (dict of torch.Tensor):        Time domain signals. 
        old_snr (float or torch.Tensor): old network SNR. If None it will be computed.
        new_snr (float or torch.Tensor): new network SNR
        kwargs:                          additional arguments to pass to the optimal_snr function. 
                                         (e.g. the sampling frequency to compute the fft)
    
    Returns:
    --------
        hnew (dict of torch.Tensor): rescaled time domain signals
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
                    
        
    for det in h:
        h_o[det] *= (new_snr/old_snr)
        
        
    return h_o