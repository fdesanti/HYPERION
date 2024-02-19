""" Normalizing Flow implementation """



import torch
import torch.nn as nn
from tqdm import tqdm
from time import time

from astropy.time import Time
from astropy.units.si import sday


class Flow(nn.Module):
    """Class that manages the flow model"""
    
    
    def __init__(self,
                 base_distribution,
                 transformation,
                 prior_metadata : dict = None,
                 embedding_network : nn.Module = None, 
                 configuration: dict = None):
        
        super(Flow, self).__init__()
        
        self.base_distribution = base_distribution
        
        self.transformation    = transformation
        
        self.embedding_network = embedding_network
        
        self.prior_metadata = prior_metadata
        
        
        self.configuration = configuration
           
        return

    @property
    def latex_labels(self):
        'e0', 'p_0', 'distance', 'time_shift', 'polarization', 'inclination', 'dec', 'ra'
        converter = {'m1':'$m_1$', 'm2':'$m_2$', 'M_tot':'$M$ $[M_{\odot}]$', 'q':'$q$','M_chirp':'$\mathcal{M}$' ,'e0':'$e_0$', 'p_0':'$p_0$', 'distance':'$d_L$ [Mpc]',
                     'time_shift':'$\delta t_p$ [s]', 'polarization':'$\psi$', 'inclination':'$\iota$', 'dec': '$\delta$', 'ra':'$\\alpha$'}
        return [converter[par_name] for par_name in self.inference_parameters ]

    @property
    def inference_parameters(self):
        if not hasattr(self, '_inference_parameters'):
            self._inference_parameters = self.prior_metadata['inference_parameters']
     
        return self._inference_parameters
    
    
    @property
    def priors(self):
        if not hasattr(self, '_priors'):
            self._priors = self.prior_metadata['priors']
        return self._priors

    @property
    def means(self):
        if not hasattr(self, '_means'):
            self._means = self.prior_metadata['means']
        return self._means
 
    @property
    def stds(self):
        if not hasattr(self, '_stds'):
            self._stds = self.prior_metadata['stds']
        return self._stds

    
        
    
    
    @property
    def reference_time(self):
        if not hasattr(self, '_reference_time'):
            self._reference_time = prior_metadata['reference_gps_time']
        return self._reference_time
    
        
    
    
    
    

    def log_prob(self, inputs, strain, evidence=False):
        """computes the loss function"""
        
        
        embedded_strain = self.embedding_network(strain)
        
        
        if evidence:
            embedded_strain = torch.cat([embedded_strain for _ in range(inputs.shape[0])], dim = 0)

        transformed_samples, logabsdet = self.transformation(inputs, embedded_strain)  #makes the forward pass 
        
        log_prob = self.base_distribution.log_prob(transformed_samples)
        
        return log_prob + logabsdet 
    
     

        
    def sample(self, num_samples, strain, batch_size = 50000, restrict_to_bounds=False, event_time = None, verbose = True, return_log_prob=False):
        start = time()
        
        samples = []
        
        embedded_strain = self.embedding_network(strain)
        nsteps = num_samples//batch_size if num_samples>=batch_size else 1
        batch_samples = batch_size if num_samples>batch_size else num_samples
        
        for _ in tqdm(range(nsteps), disable=False if verbose else True):
    
            prior_samples = self.base_distribution.sample(batch_samples)

            flow_samples, inverse_logabsdet = self.transformation.inverse(prior_samples, embedded_strain)
            
            samples.append(flow_samples)
        samples = torch.cat(samples, dim = 0)
        
        if return_log_prob:
            log_posterior = self.base_distribution.log_prob(samples) - inverse_logabsdet 
            #log_posterior = self.log_prob(samples, strain, evidence=True) 
            
            std = self.prior_metadata['stds']
            log_std = torch.sum(torch.log(torch.tensor([std[p][0] for p in self.inference_parameters])))
            log_posterior -= log_std
            

        processed_samples_dict = self._post_process_samples(samples, restrict_to_bounds, event_time) 

        #processed_samples_df = pd.DataFrame.from_dict(processed_samples_dict)

        end=time()
        if verbose:
            print(f"---> Sampling took {end-start:.3f} seconds")
        
        
        return processed_samples_dict#, log_posterior
    
    
    def _post_process_samples(self, flow_samples, restrict_to_bounds, event_time = None):
        #flow_samples must be transposed to have shape [N posterior parameters, N samples]
        flow_samples = flow_samples.T
        
        processed_samples_dict = dict()
        
        #de-standardize samples
        for i, name in enumerate(self.inference_parameters):
            #processed_samples = self.priors[name].de_standardize(flow_samples[i])
            processed_samples = flow_samples[i]*self.stds[name] + self.means[name]
            
            processed_samples_dict[name] = processed_samples.unsqueeze(1)#.cpu().numpy()

        #correct right ascension
        if event_time is not None:
            ra = processed_samples_dict['ra']
            ra_corrected = self._ra_shift_correction(ra, event_time)
            processed_samples_dict['ra'] = ra_corrected

        if restrict_to_bounds:
            num_samples = samples.shape[1]
            processed_samples_dict = self.restrict_samples_to_bounds(processed_samples_dict, num_samples)

    

        return processed_samples_dict


    def restrict_samples_to_bounds(self, processed_samples_dict, num_samples):
        restricted_samples_dict = dict()
        total_mask = torch.ones(num_samples, dtype=torch.bool)
        bounds = self.prior_metadata['parameters_bounds']

        #for name in ['dec']:
        for name in self.inference_parameters:
            #print(name, bounds[name]['min'], bounds[name]['max'])
            total_mask *= ((processed_samples_dict[name]<=bounds[name]['max']) * (processed_samples_dict[name]>=bounds[name]['min']))

        for name in self.inference_parameters:
            restricted_samples_dict [name] = processed_samples_dict[name][total_mask]
        return restricted_samples_dict, total_mask



    def _ra_shift_correction(self, ra_samples, event_time):
        """corrects ra shift due to Earth rotation with GMST (from pycbc.detector code)"""
        
        
        reference_Time = Time(self.reference_time, format="gps", scale="utc")
        #event_Time     = Time(event_time, format="gps", scale="utc")
        GMST_reference = reference_Time.sidereal_time("mean", "greenwich").rad
        
        dphase = (event_time - self.reference_time) / sday.si.scale * (2.0 * torch.pi)
        correction = (-GMST_reference + dphase) % (2.0 * torch.pi)
                
        return (ra_samples + correction) % (2*torch.pi)
    



        
        
        
        
        
        
        
        
        
    
    
