import torch
import pandas as pd

from pathlib import Path
from tensordict import TensorDict
from bilby.gw.result import CBCResult

from .flow import build_flow
from ..inference import ImportanceSampling


class PosteriorSampler():
    
    def __init__(self, 
                 flow                  = None,
                 flow_checkpoint_path  = None,
                 waveform_generator    = None,
                 num_posterior_samples = 10000,
                 output_dir            = None,
                 device                = 'cpu'):
        
        #building flow model
        if flow is not None:
            self.flow = flow.to(device).eval()
        else:
            self.flow = build_flow(checkpoint_path=flow_checkpoint_path).to(device).eval()

        self.prior_metadata = self.flow.prior_metadata
        self.num_posterior_samples = num_posterior_samples
        
        #set up importance sampler
        if waveform_generator:
            self.waveform_generator = waveform_generator
            self.IS = ImportanceSampling(flow                  = self.flow, 
                                         device                = device,
                                         waveform_generator    = waveform_generator,
                                         num_posterior_samples = num_posterior_samples)
        #other attributes
        if output_dir:
            self.output_dir = output_dir  
        else:
            self.output_dir = Path(flow_checkpoint_path).parent.absolute()

        self.device = device
        return
    
    #@property 
    def latex_labels(self, parameters=None): 
        converter = {'m1'                 : '$m_1$ $[M_{\odot}]$', 
                     'm2'                 : '$m_2$ $[M_{\odot}]$',
                     'M'                  : '$M$ $[M_{\odot}]$',
                     'q'                  : '$q$',
                     'Mchirp'             : '$\mathcal{M}$ $[M_{\odot}]$',
                     'm1_source'          : '$m_1^{source}$ $[M_{\odot}]$',
                     'm2_source'          : '$m_2^{source}$ $[M_{\odot}]$',
                     'M_source'           : '$M^{source}$ $[M_{\odot}]$',
                     'Mchirp_source'      : '$\mathcal{M}^{source}$ $[M_{\odot}]$',
                     'e0'                 : '$e_0$',
                     'p_0'                : '$p_0$',
                     'distance'           : '$d_L$ [Mpc]',
                     'time_shift'         : '$\delta t_p$ [s]',
                     'tcoal'              : '$t_{coal}$ [s]',
                     'polarization'       : '$\psi$',
                     'inclination'        : '$\iota$',
                     'dec'                : '$\delta$',
                     'ra'                 : '$\\alpha$',
                     'H_hyp'              : '$\mathcal{H}_{hyp}$',
                     'r_hyp'              : '$r_{hyp}$',
                     'j_hyp'              : '$j_{hyp}$',
                     'coalescence_angle'  : '$\phi$',
                     'luminosity_distance': '$d_L$ [Mpc]',
                     }
        _latex_labels = {}
        if parameters is None: parameters = self.inference_parameters
        for par_name in parameters:
            _latex_labels[par_name] = converter[par_name] if par_name in converter else par_name
        return _latex_labels

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
    
    def to_bilby(self, posterior=None, **kwargs):
        """Export sampler results to a bilby CBC result object."""
        
        if posterior is None:
            posterior = self.posterior
       
        if isinstance(posterior, TensorDict):
            posterior= dict(posterior.cpu())
        
        parameter_keys = list(posterior.keys())

        bilby_kwargs = {'posterior'            : pd.DataFrame.from_dict(posterior),
                        'search_parameter_keys': parameter_keys,
                        'parameter_labels'     : list(self.latex_labels(parameter_keys).values()),
                        'outdir'               : self.output_dir,
                        #'injection_parameters' : injection_parameters, 
                        }
        bilby_kwargs.update(kwargs)
        
        return CBCResult(**bilby_kwargs)
    
    def plot_skymap(self, posterior=None, **skymap_kwargs):
        """Wrapper to Bilby plot skymap method."""
        
        bilby_result = self.to_bilby(posterior=posterior)
        return bilby_result.plot_skymap(**skymap_kwargs)
    
    def plot_corner(self, posterior=None, injection_parameters=None, figname=None, **corner_kwargs):
        """Wrapper to Bilby plot corner method."""
        
        bilby_result = self.to_bilby(posterior=posterior, 
                                     injection_parameters=injection_parameters)
        
        parameter_keys = list(bilby_result.posterior.keys())
        
        fontsize_kwargs = {'fontsize': 20}
        default_corner_kwargs = {'title_kwargs'   : fontsize_kwargs, 
                                 'label_kwargs'   : fontsize_kwargs,
                                 'labels'         : list(self.latex_labels(parameter_keys).values()),
                                 'plot_density'   : False,
                                 'plot_datapoints': True}
        
        #update corner kwargs with input arguments
        default_corner_kwargs.update(corner_kwargs)
        figname = str(self.output_dir) + '/corner_plot.png' if figname is None else figname
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
        
        num_samples = sampling_kwargs.get('num_samples', self.num_posterior_samples)
        sampling_kwargs.update({'num_samples': num_samples})
        
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
        self.IS_results = self.IS.compute_model_evidence(**importance_sampling_kwargs)
        
        return self.IS_results['stats']['weights'], self.IS_results['stats']['valid_samples']
    
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
    
    
    def plot_reconstructed_waveform(self, whitened_strain, asd, num_wvf=None, posterior=None, CL=90, **kwargs):
        from ..simulations import GWDetectorNetwork, WhitenNet
        
        if posterior is None:
            posterior = self.posterior
        
        if not 'j_hyp' in posterior.keys():
            posterior['j_hyp'] = 4.0 * torch.ones_like(posterior['ra']) 
        
        if not 'polarization' in posterior.keys():
            posterior['polarization'] = torch.zeros_like(posterior['ra'])

        if num_wvf is None:
            num_wvf = len(posterior)
        
        # select random samples from the posterior
        print(f'[INFO]: Selecting {num_wvf} random samples from the posterior.')
        sub_posterior = {}
        for key in posterior.keys():
            i = torch.multinomial(torch.ones(len(posterior)), num_wvf, replacement=False)
            sub_posterior[key] = posterior[key][i]
        #posterior = TensorDict.from_dict(sub_posterior).to(self.device)

        #posterior = posterior[i]
            
        #get waveform samples
        projected_template, tcoal = self.waveform_generator(posterior, project_onto_detectors=True)
        projected_template = TensorDict.from_dict(projected_template).to(self.device)
        for det in projected_template.keys():
            projected_template[det] /= posterior['luminosity_distance'].unsqueeze(-1)
        
        #compute time_shifts
        time_shift = self.waveform_generator.det_network.time_delay_from_earth_center(posterior['ra'], 
                                                                                      posterior['dec'])
        tcoal_diff  = posterior['tcoal'].median().to(self.device)-tcoal.squeeze(1).to(self.device)
        
        for ifo in time_shift:
            time_shift[ifo] += tcoal_diff
        
        time_shift = TensorDict.from_dict(time_shift).to(self.device).unsqueeze(-1)
        
        whitener = WhitenNet(duration = self.waveform_generator.duration,
                             fs       = self.waveform_generator.fs,
                             device   = self.device)
        
        #whiten_kwargs
        whiten_kwargs = kwargs.get('whiten_kwargs', {})
        print(projected_template)
        whitened_waveform = whitener(h          = projected_template, 
                                     asd        = asd,
                                     time_shift = time_shift, 
                                     method     = 'gwpy',
                                     normalize  = True, 
                                     add_noise  = False)
        
        time = torch.linspace(-self.waveform_generator.duration/2, 
                              self.waveform_generator.duration/2, 
                              int(self.waveform_generator.duration*self.waveform_generator.fs)
                             ).cpu().numpy()
        
        
        
        import matplotlib.pyplot as plt
        import numpy as np
        plt.figure(figsize=(20, 15))
        for i, det in enumerate(self.waveform_generator.det_network.detectors):
            plt.subplot(3, 1, i+1)
            plt.plot(time, whitened_strain[det].cpu().numpy(), alpha=0.5)
            
            wvf = whitened_waveform[det].cpu().numpy()
            
            
            percentiles = [(100-CL)/2, 100-(100-CL)/2]
            median = np.percentile(whitened_waveform[det].cpu().numpy(), 50, axis=0)
            low  = np.percentile(whitened_waveform[det].cpu().numpy(), percentiles[0], axis=0)
            high = np.percentile(whitened_waveform[det].cpu().numpy(), percentiles[1], axis=0)
            plt.fill_between(time, low, high, color='r', alpha=0.5)
            
            plt.plot(time, median, linestyle = '--', color='k', alpha=0.8)


            #plt.plot(time, median, color='r', alpha=0.5)
            '''
            for wvf_sample in wvf:
                print(wvf_sample, wvf_sample.shape)
                plt.plot(time, wvf_sample, color='r', alpha=0.5)
            '''
            
            plt.title(det)           
        
        plt.savefig(f'{self.output_dir}/reconstructed_waveform_{CL}CL.png', dpi=200)
        plt.show()
        
        return 
        
            
        
                
        



        
        
        
        
    
    
    
    
