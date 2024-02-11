""" Normalizing Flow implementation """

from time import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from astropy.time import Time
from astropy.units.si import sday

#torch.backends.cudnn.allow_tf32 = False

class Flow(nn.Module):
    """Class that manages the flow model"""
    
    
    def __init__(self,
                 prior,
                 transform,
                 model_hyperparams : dict = None,
                 configuration     : dict = None,
                 embedding_network : nn.Module = None):
        super(Flow, self).__init__()
        
        self._prior     = prior
        
        
        self._transform = transform
        
        self._embedding_network = embedding_network
        

        self.model_hyperparams = model_hyperparams
        self.configuration = configuration
           
        return

    @property
    def latex_labels(self):
        'e0', 'p_0', 'distance', 'time_shift', 'polarization', 'inclination', 'dec', 'ra'
        converter = {'m1':'$m_1$', 'm2':'$m_2$', 'M_tot':'$M$ $[M_{\odot}]$', 'q':'$q$','M_chirp':'$\mathcal{M}$' ,'e0':'$e_0$', 'p_0':'$p_0$', 'distance':'$d_L$ [Mpc]',
                     'time_shift':'$\delta t_p$ [s]', 'polarization':'$\psi$', 'inclination':'$\iota$', 'dec': '$\delta$', 'ra':'$\\alpha$'}
        return [converter[par_name] for par_name in self.model_hyperparams['parameters_names'] ]


    def log_prob(self, inputs, strain, evidence=False):
        """computes the loss function"""
        
        
        embedded_strain = self._embedding_network(strain)
        
        if evidence:
            embedded_strain = torch.cat([embedded_strain for _ in range(inputs.shape[0])], dim = 0)

        transformed_samples, logabsdet = self._transform(inputs, embedded_strain)  #makes the forward pass 
        
        log_prob = self._prior.log_prob(transformed_samples)
        
        return log_prob + logabsdet 
    
     

        
    def sample(self, num_samples, strain, batch_size = 50000, restrict_to_bounds=False, event_time = None, verbose = True, return_log_prob=False):
        start = time()
        
        samples = []
        
        embedded_strain = self._embedding_network(strain)
        nsteps = num_samples//batch_size if num_samples>=batch_size else 1
        batch_samples = batch_size if num_samples>batch_size else num_samples
        
        for _ in tqdm(range(nsteps), disable=False if verbose else True):
    
            prior_samples = self._prior.sample(batch_samples)

            flow_samples, inverse_logabsdet = self._transform.inverse(prior_samples, embedded_strain)
            
            samples.append(flow_samples)
        samples = torch.cat(samples, dim = 0)
        
        if return_log_prob:
            log_posterior = self._prior.log_prob(samples) - inverse_logabsdet 
            #log_posterior = self.log_prob(samples, strain, evidence=True) 
            
            std = self.model_hyperparams['stds']
            log_std = torch.sum(torch.log(torch.tensor([std[p][0] for p in self.model_hyperparams['parameters_names']])))
            log_posterior -= log_std
            

        processed_samples_dict = self._post_process_samples(samples.T, restrict_to_bounds, event_time) #flow_samples must be transposed to have shape [N posterior parameters, N samples]

        #processed_samples_df = pd.DataFrame.from_dict(processed_samples_dict)

        end=time()
        if verbose:
            print(f"---> Sampling took {end-start:.3f} seconds")
        
        
        return processed_samples_dict, log_posterior
    
    
    def _post_process_samples(self, samples, restrict_to_bounds, event_time = None):
        
        processed_samples_dict = dict()
        
        for i, name in enumerate(self.model_hyperparams['parameters_names']):
            mean = self.model_hyperparams['means'][name][0]
            std  = self.model_hyperparams['stds'][name][0]
            processed_samples = (samples[i]*std + mean)
            
            processed_samples_dict[name] = processed_samples.unsqueeze(1)#.cpu().numpy()

        if event_time is not None:
            ra = processed_samples_dict['ra']
            ra_corrected = self._ra_shift_correction(ra, event_time)
            processed_samples_dict['ra'] = ra_corrected

        if restrict_to_bounds:
            num_samples = samples.shape[1]
            processed_samples_dict = self.restrict_samples_to_bounds(processed_samples_dict, num_samples)

        
        if 'distance' in self.model_hyperparams['parameters_names']:
            processed_samples_dict['distance'] /= 1e6
        

        processed_samples_dict['time_shift'] /= 2


        return processed_samples_dict


    def restrict_samples_to_bounds(self, processed_samples_dict, num_samples):
        restricted_samples_dict = dict()
        total_mask = np.ones(num_samples, dtype='bool')
        bounds = self.model_hyperparams['parameters_bounds']

        #for name in ['dec']:
        for name in self.model_hyperparams['parameters_names']:
            #print(name, bounds[name]['min'], bounds[name]['max'])
            total_mask *= ((processed_samples_dict[name]<=bounds[name]['max']) * (processed_samples_dict[name]>=bounds[name]['min']))

        for name in self.model_hyperparams['parameters_names']:
            restricted_samples_dict [name] = processed_samples_dict[name][total_mask]
        return restricted_samples_dict, total_mask



    def _ra_shift_correction(self, ra_samples, event_time, reference_time=1370692818.0):
        """corrects ra shift due to Earth rotation with GMST (from pycbc.detector code)"""
        
        
        reference_Time = Time(reference_time, format="gps", scale="utc")
        #event_Time     = Time(event_time, format="gps", scale="utc")
        GMST_reference = reference_Time.sidereal_time("mean", "greenwich").rad
        
        dphase = (event_time - reference_time) / sday.si.scale * (2.0 * torch.pi)
        correction = (-GMST_reference + dphase) % (2.0 * torch.pi)
                
        return (ra_samples + correction) % (2*torch.pi)
    



        
        
        
        
        
        
        
        
        
    
    
