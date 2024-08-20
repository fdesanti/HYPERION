import torch
import pandas as pd

from pathlib import Path
from tensordict import TensorDict
from .flow import Flow, build_flow
from ..inference import ImportanceSampling
from bilby.gw.result import CBCResult


class PosteriorSampler():
    
    def __init__(self, 
                 flow = None,
                 flow_checkpoint_path=None, 
                 waveform_generator=None, 
                 num_posterior_samples=10000,
                 output_dir = None,
                 device = 'cpu'):
        
        #building flow model
        if flow is not None:
            self.flow = flow.to(device).eval()
        else:
            self.flow = build_flow(checkpoint_path=flow_checkpoint_path).to(device).eval()

        self.prior_metadata = self.flow.prior_metadata
        self.num_posterior_samples = num_posterior_samples
        
        #set up importance sampler
        if waveform_generator:
            self.IS = ImportanceSampling(flow=self.flow, 
                                         device=device,
                                         waveform_generator=waveform_generator, 
                                         num_posterior_samples=num_posterior_samples)
        #other attributes
        if output_dir:
            self.output_dir = output_dir  
        else:
            self.output_dir = Path(flow_checkpoint_path).parent.absolute()

        self.device = device
        return
    
    @property #TODO: this is a bit ugly, make it more elegant
    def latex_labels(self): 
        converter = {'m1':'$m_1$', 'm2':'$m_2$', 'M':'$M$ $[M_{\odot}]$', 'q':'$q$','M_chirp':'$\mathcal{M}$' ,'e0':'$e_0$', 'p_0':'$p_0$', 'distance':'$d_L$ [Mpc]',
                     'time_shift':'$\delta t_p$ [s]', 'polarization':'$\psi$', 'inclination':'$\iota$', 'dec': '$\delta$', 'ra':'$\\alpha$'}
        latex_labels = []
        for par_name in self.inference_parameters:
            label = converter[par_name] if par_name in converter else par_name
            latex_labels.append(label)
        return latex_labels

    @property
    def inference_parameters(self):
        return self.flow.inference_parameters
    
    @property
    def priors(self):
        return self.flow.priors

    @property
    def means(self):
        return self.flow.means
 
    @property
    def stds(self):
        return self.flow.stds

    @property
    def reference_time(self):
        return self.flow.reference_time
    
    @property
    def posterior(self):
        if not hasattr(self, '_posterior'):
            raise ValueError('Posterior has not been sampled yet. Run the "sample_posterior" method.')
        return self._posterior
    
    @posterior.setter
    def posterior(self, posterior):
        self._posterior = posterior
        
    
    @property
    def importance_weights(self):
        if not hasattr(self, '_importance_weights'):
            raise ValueError('Importance weights have not been computed yet. Run the "compute_importance_weights" method.')
        return self._importance_weights
    
    @property
    def reweighted_posterior(self):
        if not hasattr(self, '_reweighted_posterior'):
            raise ValueError('Importance reweighted posterior has not been computed yet. Run the "reweight_posterior" method.')
        return self._reweighted_posterior
    
    @reweighted_posterior.setter
    def reweighted_posterior(self, reweighted_posterior):
        self._reweighted_posterior = reweighted_posterior
    
    @property
    def BayesFactor(self):
        if not hasattr(self, '_BayesFactor'):
            raise ValueError('Bayes Factor has not been computed yet. Run the "compute_BayesFactor" method.')
        return self._BayesFactor
    
    def to_bilby(self, posterior=None):
        """Export sampler results to a bilby CBC result object."""
        
        if posterior is None:
            posterior = self.posterior
       
        if isinstance(posterior, TensorDict):
            posterior= dict(posterior.cpu())

        bilby_kwargs = {'posterior': pd.DataFrame.from_dict(posterior),
                        'search_parameter_keys': self.inference_parameters,
                        'parameter_labels': self.latex_labels,
                        'outdir': self.output_dir}
        
        return CBCResult(**bilby_kwargs)
    
    def plot_skymap(self, bilby_posterior=None, **skymap_kwargs):
        """Wrapper to Bilby plot skymap method."""
        
        bilby_result = self.to_bilby(bilby_posterior)
        return bilby_result.plot_skymap(**skymap_kwargs)
    
    def plot_corner(self, posterior=None, injection_parameters=None, **corner_kwargs):
        """Wrapper to Bilby plot corner method."""

        bilby_result = self.to_bilby(posterior)

        fontsize_kwargs = {'fontsize': 20}
        default_corner_kwargs = {'title_kwargs':fontsize_kwargs, 'label_kwargs':fontsize_kwargs, 
                                 'labels':self.latex_labels, 'plot_density':False, 
                                 'plot_datapoints':True}
        
        #update corner kwargs with input arguments
        default_corner_kwargs.update(corner_kwargs)
        figname = str(self.output_dir) + '/corner_plot.png'
        return bilby_result.plot_corner(filename=figname, truth=injection_parameters, **default_corner_kwargs)
    

    def sample_posterior(self, **sampling_kwargs):
        """
        Samples posterior from the flow model.
        
        Note:
        -----
            For a list of the input arguments see the documentation of the Flow.sample method.
            
        Returns:
        --------
            posterior: (dict)
                A dictionary containing the samples from the posterior distribution.
        """
        #TODO: better manage options
        with torch.inference_mode():
            self.posterior = self.flow.sample(**sampling_kwargs)
        return self.posterior
            
    def sample_importance_weights(self, **importance_sampling_kwargs):
        """
        Sample importance weights using the importance sampler.
        
        Returns:
        --------
            importance_weights: (torch.Tensor)
                A tensor containing the importance weights.
                
        Note:
        -----
            For a list of the input arguments see the documentation of the ImportanceSampling.compute_model_evidence method.

        
        """
        self._IS_results = self.IS.compute_model_evidence(**importance_sampling_kwargs)
        
        return self._IS_results['stats']['weights'], self._IS_results['stats']['valid_samples']
    
    def reweight_posterior(self, posterior=None, num_samples=None, importance_weights=None, importance_sampling_kwargs=None):
        """
        Sample from the importance reweighted posterior.
        
        Args:
        -----
            importance_weights: (torch.Tensor)
                A tensor containing the importance weights.
        
        Returns:
        --------
            reweighted_posterior: (dict)
                A dictionary containing the reweighted samples from the posterior distribution.
        """
        
        if posterior is None:
            print('[INFO]: Using previously sampled posterior.')
            posterior = self.posterior
       
        if importance_weights is None:
            if importance_sampling_kwargs is None:
                raise ValueError('Either importance_weights or importance_sampling_kwargs must be provided.')
            importance_weights, valid_samples = self.sample_importance_weights(**importance_sampling_kwargs)
        
        #discard invalid samples in regular posterior
        self.posterior = posterior[valid_samples]
        if not num_samples:
            num_samples = len(self.posterior)
        
        reweighted_indexes = torch.multinomial(importance_weights, num_samples, replacement=True)
        self.reweighted_posterior = self.posterior[reweighted_indexes]
        
        
        return self.reweighted_posterior
        



        
        
        
        
    
    
    
    
