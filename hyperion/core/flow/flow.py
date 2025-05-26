""" Normalizing Flow implementation """

import torch
import torch.nn as nn

from tqdm import tqdm
from time import time

from astropy.time import Time
from astropy.units.si import sday

from ..types import TensorSamples
from ..utilities import HYPERION_Logger

log = HYPERION_Logger()

class Flow(nn.Module):
    """
    Class that manages the Normalizing Flow model
    
    Args:
        base_distribution (BaseDistribution) : Instance of one of the hyperion.core.distributions.flow_base subclasses
        transformation    (CouplingTransform): The (coupling) transformation of the model
        embedding_network         (nn.Module): (optional) Embedding network for the strain data. (Default: None)
        metadata                       (dict): Metadata of the prior distribution.
        prior_metadata                 (dict): (optional) Prior metadata. If provided will be used to update the metadata
    """

    def __init__(self,
                 base_distribution,
                 transformation,
                 embedding_network: nn.Module = None,
                 metadata         : dict = {}, 
                 prior_metadata   : dict = None
                 ):
        
        super(Flow, self).__init__()
        
        self.base_distribution = base_distribution
        self.transformation    = transformation
        self.metadata          = metadata

        if prior_metadata is not None:
            assert isinstance(prior_metadata, dict), 'Please provide a dictionary with the prior metadata.'
            self.metadata['prior_metadata'] = prior_metadata

        else:
            assert isinstance(metadata, dict), "Please provide 'metadata' as a dictionary."
            assert 'prior_metadata' in metadata, "Please make sure metadata contains a 'prior_metadata' key."

        if embedding_network is not None:
            self.embedding_network = embedding_network

    @property
    def prior_metadata(self):
        return self.metadata['prior_metadata']
    
    @property
    def inference_parameters(self):
        """List of parameters to be inferred"""
        if not hasattr(self, '_inference_parameters'):
            if 'inference_parameters' in self.metadata:
                self._inference_parameters = self.metadata['inference_parameters']
            else:
                self._inference_parameters = self.prior_metadata['inference_parameters']
        
        return self._inference_parameters
    
    @property
    def priors(self):
        """Priors of the simulation"""
        if not hasattr(self, '_priors'):
            self._priors = self.metadata['prior_metadata']['priors']
        return self._priors

    @property
    def means(self):
        """Mean of each parameter prior"""
        if not hasattr(self, '_means'):
            self._means = self.metadata['prior_metadata']['means']
        return self._means
 
    @property
    def stds(self):
        """Standard deviation of each parameter prior"""
        if not hasattr(self, '_stds'):
            self._stds = self.metadata['prior_metadata']['stds']
        return self._stds
    
    
    def log_prob(self, inputs, condition_data=None):
        r"""
        Computes the log probability of the input samples

        .. math::
            \log q_\phi(\theta) = \log p(z) + \log \left| \det \left( \frac{d f_\phi (\theta)}{d\theta} \right) \right|

        Args:
        
            inputs         (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            condition_data (torch.Tensor): (Optional) Data on which condition the flow. (Default: None)
        """

        # we set the embedding to None if the embedding network is not defined
        if condition_data is not None and hasattr(self, 'embedding_network'):
            embedded_data = self.embedding_network(condition_data)
        else:
            embedded_data = None
        
        #transform theta --> z
        transformed_samples, logabsdet = self.transformation(inputs, embedded_data)  #makes the forward pass 
        
        #compute the log probability of the base distribution
        log_prob = self.base_distribution.log_prob(transformed_samples, embedded_data)
        
        return log_prob + logabsdet 
    
     
    def sample(self, num_samples, condition_data=None, batch_size=5e4, restrict_to_bounds=False, post_process=True, verbose=True, return_log_prob=False):
        r"""
        Sample the Flow posterior distribution.

        .. math::
            \theta = f_\phi^{-1}(z) \text{ where } z \sim p(z)

        Args:
            num_samples         (int): Number of samples to draw from the posterior
            condition  (torch.Tensor): (Optional) Data on which condition the flow. (Default: None)
            batch_size          (int): Batch size for the sampling process. (Default: 50k)
            restrict_to_bounds (bool): If True, restrict the samples to the prior bounds. (Default: False)
            post_process       (bool): If True, post-process the samples. (Default: True)
            verbose            (bool): If True, print the sampling time. (Default: True)
            return_log_prob    (bool): If True, return the log probability of the posterior samples. (Default: False)
        
        Returns:
            tuple: Tuple containing
                - **samples** (TensorSamples): The posterior samples
                - **log_posterior** (torch.Tensor): Log probability of the posterior samples. (Returned only if ``return_log_prob`` is True)
        """
        #take the start time
        start = time()

        #cast correctly
        num_samples = int(num_samples)
        batch_size  = int(batch_size)
        
        #embedding strain
        if condition_data is not None and hasattr(self, 'embedding_network'):
            embedded_data = self.embedding_network(condition_data)
        else:
            embedded_data = None
        
        samples              = []
        nsteps               = num_samples // batch_size if num_samples>=batch_size else 1
        batch_samples        = batch_size if num_samples > batch_size else num_samples
        disable_progress_bar = True if not verbose or nsteps == 1 else False

        #sampling
        for _ in tqdm(range(nsteps), disable=disable_progress_bar):
            #sample from the latent base distribution
            prior_samples = self.base_distribution.sample(batch_samples, embedded_data)
            #transform back to the original space
            flow_samples, inverse_logabsdet = self.transformation.inverse(prior_samples, embedded_data)
            samples.append(flow_samples)
        
        samples = torch.cat(samples, dim = 0)          

        #post-processing
        if post_process:
            processed_samples = self.post_process_samples(samples, restrict_to_bounds) 
        else:
            processed_samples_dict = dict()
            for i, name in enumerate(self.inference_parameters):
                processed_samples_dict[name] = samples[:,i]
                processed_samples = TensorSamples.from_dict(processed_samples)

        end=time()

        if verbose:
            log.info(f"Sampling took {end-start:.3f} seconds")
        
        if return_log_prob:
            log_posterior = self.base_distribution.log_prob(samples, embedded_data) - inverse_logabsdet 
            #log_posterior = self.log_prob(samples, strain, evidence=True) 
            
            log_std = torch.sum(torch.log(torch.tensor([self.stds[p] for p in self.inference_parameters])))
            log_posterior -= log_std
            return processed_samples, log_posterior
        
        else:
            return processed_samples
    
    def post_process_samples(self, flow_samples, restrict_to_bounds=False):
        """
        Post-process the samples by applyng de-standardization.
        If restrict_to_bounds is True, the samples are restricted to the prior bounds.
        """
        #flow_samples must be transposed to have shape [N posterior parameters, N samples]
        flow_samples = flow_samples.T

        processed_samples_dict = dict()
        
        #de-standardize samples
        for i, name in enumerate(self.inference_parameters):
            #processed_samples = self.priors[name].de_standardize(flow_samples[i])
            processed_samples = flow_samples[i]*self.stds[name] + self.means[name]
            
            processed_samples_dict[name] = processed_samples#.cpu().numpy()

        if restrict_to_bounds:
            num_samples = flow_samples.shape[1]
            processed_samples_dict = self.restrict_samples_to_bounds(processed_samples_dict, num_samples)
        
        return TensorSamples.from_dict(processed_samples_dict)


    def restrict_samples_to_bounds(self, processed_samples_dict, num_samples):
        """Restrict the samples to the prior bounds by masking out the samples that are outside the bounds"""

        restricted_samples_dict = dict()
        device = processed_samples_dict[self.inference_parameters[0]].device 
        total_mask = torch.ones(num_samples, dtype=torch.bool, device=device)

        for name in self.inference_parameters:
            try: 
                if "bounds" in self.prior_metadata:
                    min_b, max_b = self.prior_metadata['bounds'][name]
                else:
                    bounds = self.prior_metadata['parameters'][name]['kwargs']
                    min_b = eval(bounds['minimum']) if isinstance(bounds['minimum'], str) else bounds['minimum']
                    max_b = eval(bounds['maximum']) if isinstance(bounds['maximum'], str) else bounds['maximum']
                
                total_mask *= ((processed_samples_dict[name]<=max_b) * (processed_samples_dict[name]>=min_b))
            
            except Exception as e:
                log.error(f"Could not restrict samples for {name} to bounds: due to {e}")
                continue

        for name in self.inference_parameters:
            restricted_samples_dict [name] = processed_samples_dict[name][total_mask]
        return restricted_samples_dict


        
        
class GWFlow(Flow):
    """
    Class that manages the Normalizing Flow model for gravitational wave signals.
    
    Args:
        base_distribution (BaseDistribution) : Instance of one of the hyperion.core.distributions.flow_base subclasses
        transformation    (CouplingTransform): The (coupling) transformation of the model
        embedding_network         (nn.Module): (optional) Embedding network for the strain data. (Default: None)
        metadata                       (dict): Metadata of the prior distribution.
        prior_metadata                 (dict): (optional) Prior metadata. If provided will be used to update the metadata
    """
    
    def __init__(self,
                 base_distribution,
                 transformation,
                 embedding_network: nn.Module = None,
                 metadata         : dict = {}, 
                 prior_metadata   : dict = None
                 ):
        
        super(GWFlow, self).__init__(base_distribution, transformation, embedding_network, metadata, prior_metadata)
        
    @property
    def reference_time(self):
        """Reference GPS time for the simulation"""
        if not hasattr(self, '_reference_time'):
            self._reference_time = self.metadata['reference_gps_time']
        return self._reference_time
    
    def ra_shift_correction(self, ra_samples, event_time):
        r"""
        Correct the right ascension (RA) samples by shifting them according to the event time.
        This is done by computing a correction factor based on the Greenwich Mean Sidereal Time (GMST) at the event time.
        The correction is applied as follows:
        
        .. math::

            \text{RA} \rightarrow \text{RA} + \alpha \mod 2\pi

        where 

        .. math::

            \alpha = \text{GMST}_{\text{event}} - \text{GMST}_{\text{reference}} \mod 2\pi

        """
        reference_Time = Time(self.reference_time, format="gps", scale="utc")
        event_Time     = Time(event_time, format="gps", scale="utc")
        GMST_event     = event_Time.sidereal_time("mean", "greenwich").rad
        GMST_reference = reference_Time.sidereal_time("mean", "greenwich").rad
        
        correction = (GMST_event - GMST_reference) % (2*torch.pi)

        log.info(f"RA correction: {correction:.3f} rad")
                
        return (ra_samples + correction) % (2*torch.pi)
    

    def log_prob(self, inputs, strain=None):
        r"""
        Computes the log probability of the input samples

        .. math::
            \log q_\phi(\theta) = \log p(z) + \log \left| \det \left( \frac{d f_\phi (\theta)}{d\theta} \right) \right|

        Args:
            inputs (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            strain (torch.Tensor): (Optional) Tensor of shape [N, C, L] where N is the number of samples, C is the number of channels and L is the length of the strain data. (Default: None)
        """
        return super().log_prob(inputs, condition_data=strain)
    
    def sample(self, num_samples, strain, event_time=None, return_log_prob=False, **sample_kwargs):
        r"""
        Sample the Flow posterior distribution.

        .. math::
            \theta = f_\phi^{-1}(z) \text{ where } z \sim p(z)

        Args:
            num_samples          (int): Number of samples to draw from the posterior
            strain      (torch.Tensor): Tensor of shape [N, C, L] where N is the number of samples, C is the number of channels and L is the length of the strain data. (Default: None)
            event_time         (float): GPS time of the event. Used to correct the right ascension (RA), if inferred. (Default: None)
            return_log_prob     (bool): If True, return the log probability of the posterior samples. (Default: False)
            **sample_kwargs     (dict): Additional keyword arguments to pass to the sample method of the base distribution.
        
        Returns:
            tuple: Tuple containing
                - **samples** (TensorSamples): The posterior samples
                - **log_posterior** (torch.Tensor): Log probability of the posterior samples. (Returned only if ``return_log_prob`` is True)
        """

        #first sample from the flow
        result = super().sample(num_samples     = num_samples,
                                condition_data  = strain,
                                return_log_prob = return_log_prob,
                                **sample_kwargs)

        #if return_log_prob is True, the result is a tuple of (samples, log_posterior)
        if return_log_prob:
            samples, log_posterior = result
        else:
            samples = result
            log_posterior = None
        
        #apply the RA shift correction if needed
        if event_time is not None and 'ra' in self.inference_parameters:
            samples['ra'] = self.ra_shift_correction(samples['ra'], event_time)
        
        #return the samples
        if return_log_prob:
            return samples, log_posterior
        else:
            return samples