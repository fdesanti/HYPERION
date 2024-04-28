
"""This module contains a class to generate and save a training dataset into hdf5 files"""

import yaml
import torch
import torch.nn.functional as F
from torch.distributions import Gamma

from ..core.fft import *
from ..config import CONF_DIR
from ..core.distributions import MultivariatePrior, prior_dict_
from ..core.distributions import q_uniform_in_components as q_prior

from ..simulations import WhitenNet

from ..simulations.sim_utils import (optimal_snr, 
                                     matched_filter_snr, 
                                     rescale_to_network_snr, 
                                     network_optimal_snr)


class DatasetGenerator:
    """
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    
    Given a specified prior and a waveform generator it generates as output a tuple
    (parameters, whitened_strain)

    """

    def __init__(self, 
                 waveform_generator, 
                 asd_generators, 
                 det_network,
                 prior_filepath  = None, 
                 signal_duration = 1, 
                 noise_duration  = 2, 
                 batch_size      = 512,
                 device          = 'cpu',
                 random_seed     = None,
                 inference_parameters = None,
                 ):
        """
        Constructor.
        """
        
        self.waveform_generator   = waveform_generator
        self.asd_generator        = asd_generators
        self.batch_size           = batch_size
        self.signal_duration      = signal_duration
        self.noise_duration       = noise_duration
        self.device               = device
        self.inference_parameters = inference_parameters
        self.det_network          = det_network
        #print('>>>>>>><',self.inference_parameters)

        #set up self random number generator
        self.rng  = torch.Generator(device)
        if not random_seed:
            random_seed = torch.randint(0, 2**32, (1,)).item()
        self.rng.manual_seed(random_seed)
        self.seed = random_seed


        #load prior        
        self._load_prior(prior_filepath)     

        self.WhitenNet = WhitenNet(duration=waveform_generator.duration, 
                                   fs= waveform_generator.fs, 
                                   device=device,
                                   rng=self.rng)
        return
    
    

    @property
    def means(self):
        if not hasattr(self, '_means'):
            self._means = dict()
            for p in self.inference_parameters:
                self._means[p] = float(self.full_prior[p].mean)
        return self._means
    
    @property
    def stds(self):
        if not hasattr(self, '_stds'):
            self._stds = dict()
            for p in self.inference_parameters:
                self._stds[p] = float(self.full_prior[p].std)
        return self._stds

    @property
    def inference_parameters(self):
        return self._infer_pars
    @inference_parameters.setter
    def inference_parameters(self, name_list):
        self._infer_pars = name_list


    def _load_prior(self, prior_filepath):
        """
        Load the prior distributions specified in the json prior_filepath:
        if no filepath is given, the default one stored in the config dir will be used
        
        This function first reads the json file, then store the prior as a hyperion's MultivariatePrior instance. 
        Prior's metadata are stored as well. Metadata also contains the list of the inference parameters 
        The reference_time of the GWDetector instances is finally updated to the value set in the prior

        """

        #load extrinsic prior
        with open(prior_filepath, 'r') as f:
            prior_conf = yaml.safe_load(f)
            intrinsic_prior_conf = prior_conf['parameters']['intrinsic']
            extrinsic_prior_conf = prior_conf['parameters']['extrinsic']

        #intrinsic/extrinsic priors (used for sampling)
        self.intrinsic_prior = MultivariatePrior(intrinsic_prior_conf, device=self.device, seed=self.seed)
        self.extrinsic_prior = MultivariatePrior(extrinsic_prior_conf, device=self.device, seed=self.seed)

        #Construct a full prior combining intrinsic and extrinsic priors
        self.full_prior = self.intrinsic_prior.priors.copy()
        self.full_prior.update(self.extrinsic_prior.priors.copy())
        
        #construct prior_metadata dictionary
        self.prior_metadata = dict()
        self.prior_metadata['parameters'] = intrinsic_prior_conf
        self.prior_metadata['parameters'].update(extrinsic_prior_conf)
        self.prior_metadata['means'] = self.means
        self.prior_metadata['stds']  = self.stds
        self.prior_metadata['inference_parameters'] = self.inference_parameters

        #add M and q to prior dictionary
        #NB: they are not added to MultivariatePrior to avoid conflict with the waveform_generator 
        #    this is intended when the inference parameters contain parameters that are combination of the default's one
        #    (Eg. the total mass M =m1+m2 or q=m2/m1 that have no simple joint distribution) 
        #    In this way we store however the metadata (eg. min and max values) without compromising the simulation 
             
        if ('M' in self.inference_parameters) and ('q' in self.inference_parameters):
            for p in ['M', 'q']:
                self.prior[p] = prior_dict_[p](self.prior['m1'], self.prior['m2'])
                min, max = float(self.prior[p].minimum), float(self.prior[p].maximum)
                
                metadata = {'distribution':f'{p}_uniform_in_components', 'kwargs':{'minimum': min, 'maximum': max}}
                self.prior_metadata['parameters'][p] = metadata
        
        return 
    

    def _compute_M_and_q(self, prior_samples):

        #sorting m1 and m2 so that m2 <= m1
        m1 = prior_samples['m1'].mT.squeeze()
        m2 = prior_samples['m2'].mT.squeeze()
        
        m = torch.stack([m1, m2])
        if m.ndim > 1:
            m = m.T 

        m, _ = torch.sort(m)
        if m.ndim > 1:
            m = m.T
        
        m1, m2  = m[1], m[0]
        
        #m1 and m2 have shape [Nbatch]
        prior_samples['M'] = (m1+m2).reshape((self.batch_size, 1))
        prior_samples['q'] = (m2/m1).reshape((self.batch_size, 1))
        
        return prior_samples
    



      
    
    
    
    def standardize_parameters(self, prior_samples):
        """Standardize prior samples to zero mean and unit variance"""
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            standardized = self.prior[parameter].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized)
            
        out_prior_samples = torch.cat(out_prior_samples, dim=-1)
        return out_prior_samples
    


    def __getitem__(self, idx=None, return_hp_and_hc=False, add_noise=True, prior_samples=None):

        #sampling prior-
        if not prior_samples:
            prior_samples = self.multivariate_prior.sample((self.batch_size, 1))

        out_prior_samples = self._compute_M_and_q(prior_samples.copy())
        
        #generate projected waveform strain
        h = self.waveform_generator(**prior_samples, 
                                    return_hp_and_hc=return_hp_and_hc)
        
        
        if return_hp_and_hc:
            return out_prior_samples, h
        
        #apply time shift and whiten
        whitened_template = self._apply_time_shifts_and_whiten(h)
        
        #add gaussian noise
        if add_noise:
            whitened_strain = self._add_noise(whitened_template).float()
        else:
            whitened_strain = whitened_template.float()

        #standardize parameters
        out_prior_samples = self.standardize_parameters(out_prior_samples).float()
        
        
        return out_prior_samples, whitened_strain
    






