
"""This module contains a class to generate and save a training dataset into hdf5 files"""

import yaml
import torch
import torch.nn.functional as F

from tensordict import TensorDict

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
                 batch_size      = 512,
                 device          = 'cpu',
                 random_seed     = None,
                 num_preload     = 1000,
                 n_proc          = 10,
                 inference_parameters = None,
                 ):
        """
        Constructor.
        """
        
        self.waveform_generator   = waveform_generator
        self.asd_generator        = asd_generators
        self.batch_size           = batch_size
        self.device               = device
        self.inference_parameters = inference_parameters
        self.det_network          = det_network
        self.n_proc               = n_proc
    
        assert num_preload >= batch_size, 'The number of waveform to preload must be greater than batch_size'
        self.num_preload = num_preload

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
        self.prior_metadata['inference_parameters'] = self.inference_parameters

        #add M and q to prior dictionary
        #NB: they are not added to MultivariatePrior to avoid conflict with the waveform_generator 
        #    this is intended when the inference parameters contain parameters that are combination of the default's one
        #    (Eg. the total mass M =m1+m2 or q=m2/m1 that have no simple joint distribution) 
        #    In this way we store however the metadata (eg. min and max values) without compromising the simulation 
        if ('M' in self.inference_parameters) and ('q' in self.inference_parameters):
            for p in ['M', 'q']:
                self.full_prior[p] = prior_dict_[p](self.full_prior['m1'], self.full_prior['m2'])
                min, max = float(self.full_prior[p].minimum), float(self.full_prior[p].maximum)
                
                metadata = {'distribution':f'{p}_uniform_in_components', 'kwargs':{'minimum': min, 'maximum': max}}
                self.prior_metadata['parameters'][p] = metadata

        #store means and stds
        self.prior_metadata['means'] = self.means
        self.prior_metadata['stds']  = self.stds
        
        return 
    

    def _compute_M_and_q(self, prior_samples):

        #sorting m1 and m2 so that m2 <= m1
        m1, m2 = prior_samples['m1'], prior_samples['m2']
        m1, m2 = q_prior._sort_masses(m1, m2)
        
        #m1 and m2 have shape [Nbatch]
        prior_samples['M'] = (m1+m2)
        prior_samples['q'] = (m2/m1)

        return prior_samples
         
    
    
    
    def standardize_parameters(self, prior_samples):
        """Standardize prior samples to zero mean and unit variance"""
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            standardized = self.full_prior[parameter].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized)
            
        out_prior_samples = torch.cat(out_prior_samples, dim=-1)
        return out_prior_samples
    
    
    def get_idxs(self):
        if not hasattr(self, 'preloaded_wvfs'):
            raise ValueError('There are no preloaded waveforms. Please run pre_load_waveforms() first.')

        idxs = torch.arange(self.num_preload).float()
        return torch.multinomial(idxs, self.batch_size, replacement=False)
    

    def preload_waveforms(self):
        """
        Preload a set of waveforms to speed up the generation of the dataset.
        """
        
        print('[INFO] Preloading a new set of waveforms...')

        #first we sample the intrinsic parameters
        self.prior_samples = self.intrinsic_prior.sample(self.num_preload)
        
        
        if all(p in self.inference_parameters for p in ['M', 'q']):
            self.prior_samples = self._compute_M_and_q(self.prior_samples)
        

        #then we call the waveform generator
        
        hp, hc, tcoal = self.waveform_generator(self.prior_samples.to('cpu'), 
                                                n_proc=self.n_proc)
        print('[INFO] Done')

        
        #store the waveforms as a TensorDict
        wvfs = {'hp': hp, 'hc': hc}
        tcoals = {'tcoal': tcoal}

        self.preloaded_wvfs = TensorDict.from_dict(wvfs).to(self.device)
        self.tcoals = TensorDict.from_dict(tcoals).to(self.device)
        
        return


    def __getitem__(self, add_noise=True):

        idxs = self.get_idxs()

        #get the prior samples
        prior_samples = self.prior_samples[idxs].unsqueeze(1)

        #get the corresponding preloaded waveforms
        hp, hc = self.preloaded_wvfs[idxs]['hp'], self.preloaded_wvfs[idxs]['hc']

        #sample extrinsic priors
        prior_samples.update(self.extrinsic_prior.sample((self.batch_size, 1)))

        #rescale luminosity distance
        hp /= prior_samples['distance']
        hc /= prior_samples['distance']

        #project strain onto detectors
        h = self.det_network.project_wave(hp, hc, 
                                          ra=prior_samples['ra'], 
                                          dec=prior_samples['dec'], 
                                          polarization=prior_samples['polarization'])
        #compute relative time shifts
        time_shifts = self.det_network.time_delay_from_earth_center(ra=prior_samples['ra'], 
                                                                    dec=prior_samples['dec'])        
        
        for det in h.keys():
            time_shifts[det] += prior_samples['time_shift']

        #sample asd --> whiten --> add noise
        #asd = {det: self.asd_generator[det].sample(self.batch_size) for det in self.det_network.detectors}
        asd = {}
        noise = {}
        for det in self.det_network.detectors:
            asd[det], noise[det] = self.asd_generator[det].sample(self.batch_size, noise=True)
        
        whitened_strain = self.WhitenNet(h=h, 
                                         asd=asd, 
                                         noise = noise,
                                         time_shift=time_shifts, 
                                         add_noise=add_noise)

        #standardize parameters
        out_prior_samples = self.standardize_parameters(prior_samples)

        #convert to a single float tensor
        out_whitened_strain = torch.stack([whitened_strain[det] for det in self.det_network.detectors], dim=1)
        
        return out_prior_samples.float(), out_whitened_strain.float()
    






