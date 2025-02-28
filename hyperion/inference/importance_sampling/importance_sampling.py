import torch 
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import kstest
from .likelihood import  GWLikelihood
from ...core.utilities import latexify
from ...core.distributions.prior_distributions import MultivariatePrior


class IS_Priors(MultivariatePrior):
    """ 
    Wrapper to the MultivariatePrior class.
    
    Args:
    ----- 
        flow (hyperion.core.Flow): Flow model. Its prior is used to initialized the various parameter priors
        device              (str): either 'cpu' or 'cuda' to enable GPU. (Default 'cpu')       
    """
    def __init__(self, flow, device = 'cpu'):
        
        self.flow = flow        
        priors = flow.prior_metadata['parameters'].copy()
        
        if 'M' in priors and 'Mchirp' in priors:
            priors.pop('Mchirp')

        super(IS_Priors, self).__init__(priors, device=device)
    

class ImportanceSampling():    
    r"""
    Class that performs Importance Sampling to estimate the Bayes Factor:

    .. math::

        \mathcal{B} = \frac{Z_{\text{model}}}{Z_{\text{noise}}}

    where

    .. math::

        Z_{\text{model}} = \frac{1}{N} \sum \frac{p(\theta)\, p(s|\theta)}{q(\theta|s)}

    and

    .. math::

        Z_{\text{noise}} = \text{noise likelihood}
    
    Args:
    -----
        flow                (hyperion.core.flow): Flow object
        waveform_generator   (WaveformGenerator): Waveform Generator object
        device                             (str): Either 'cpu' or 'cuda'. (Default 'cpu')
        num_posterior_samples              (int): Number of posterior samples to draw
        reference_posterior_samples (TensorDict): (Optional) reference posterior samples to compare with. They are used in the FAR computation to derive a detection statistics based on Bayes Factor
    """
    def __init__(self, 
                 flow, 
                 waveform_generator, 
                 device                      = 'cpu',
                 num_posterior_samples       = 1000,
                 reference_posterior_samples = None):
        
        #assing attributes
        self.flow   = flow.to(device)
        self.priors = IS_Priors(flow, device)
        self.device = device
        self.num_posterior_samples = num_posterior_samples
        self.waveform_generator    = waveform_generator
        self.reference_posterior_samples = reference_posterior_samples

        #instantiate the likelihood
        self.likelihood = GWLikelihood(waveform_generator, device=device)
    
    @property
    def logB(self):
        if not hasattr(self, '_logB'):
            raise ValueError('The Bayes factor has not been calculated yet. Run the compute_Bayes_factor() method.')
        return self._logB
    
    @property
    def log10B(self):
        if not hasattr(self, '_log10B'):
            raise ValueError('The Bayes factor has not been calculated yet. Run the compute_Bayes_factor() method.')
        return self._log10B
    
    @property
    def det_names(self):
        return self.waveform_generator.det_network.names
    
    @property
    def N(self):
        if not hasattr(self, '_N'):
            self._N = torch.tensor(self.num_posterior_samples)
        return self._N
    
    @property
    def inference_parameters(self):
        return self.flow.inference_parameters
    
    @staticmethod
    @latexify
    def plot_IS_results(IS_results, savepath=None, **rcParams):
        """
        Produces a plot with the Importance Sampling results. 
        More specifically a scatter plot of the log flow posterior vs log prior + log likelihood
        with a colormap given by the importance weights.

        Args:
        -----
            IS_results (dict): Dictionary produced by the compute_model_evidence method
            savepath   (Path): (Optional) path to which save the plot

        Returns:
        --------
            fig (plt.fig): plot figure object 
        """
        #extract what we need
        valid_samples  = IS_results['valid_samples']
        log_prior      = IS_results['log_prior'][valid_samples]
        log_likelihood = IS_results['log_likelihood']
        log_posterior  = IS_results['log_posterior'][valid_samples]
        weights        = IS_results['weights']

        
        #update rcParams
        plt.rcParams['font.size'] = rcParams.get('fontsize', 18)

        #create the plot
        fig, ax = plt.subplots()
        ax.scatter(log_posterior.cpu().numpy(), (log_prior+log_likelihood).cpu().numpy(),  c=weights.cpu().numpy(), cmap='viridis', s=2)
        ax.set_xlabel(r'$\log q_\phi(\theta|s)$')
        ax.set_ylabel(r'$\log \,p(\theta) + \log \,\mathcal{L}(s|\theta)$')
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label(r'$w_i$')
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=200)
        plt.show()
        return fig
    
    @staticmethod
    def sample_efficiency( weights):
        r"""
        Computes the sampling efficiency of the Importance Sampling method.
        It is defined as:

        .. math::

            \text{eff} = \frac{1}{N} \left( \sum w_i \right)^2 \, / \, \sum w_i^2

        Args:
        -----
            weights (torch.tensor): Importance weights

        Returns:
        --------
            eff (float): Sampling efficiency (in percentage)
        """
        N = len(weights)
        eff = (1/N)* (weights.sum())**2 / (weights**2).sum()
        return eff*100
    
    def _noise_log_evidence(self, strain, psd):
        """
        Computes  log \( Z \) assuming the null hypothesis of having noise only data \( Z = L(s | \theta) \)
        
        Args:
        -----
            strain (dict, TensorDict): Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            psd    (dict, TensorDict): Dictionary containing interferometer Power Spectral Densities
        
        Returns:
        --------
            logL_noise (torch.Tensor): Log noise Likelihood
        """ 
        return self.likelihood.noise_log_Likelihood(strain, psd)
    
    def _sample_flow_posterior(self, whitened_strain, event_time):
        """
        Samples the posterior for the given stretch of data
        
        Args:
        -----
            whitened_strain (dict, TensorDict): Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            event_time                 (float): Strain (central) GPS time. It is used to correct the right ascension prediction
                
        Returns:
        --------
            theta        (TensorSamples): Posterior samples
            log_posterior (torch.Tensor): Log flow posterior defined as log \(q(\theta| s) = log N(f^{-1}(u)) + log J(f^{-1}(u))\)
            medians               (list): Medians of the posterior samples
            ks_stat              (float): Kolmogorov-Smirnov statistic comparing posterior samples to reference samples. If no reference samples are provided, it returns 'None'
        """
        torch_strain = torch.stack([whitened_strain[det] for det in self.det_names]).unsqueeze(0)
        
        #sample flow posterior
        with torch.inference_mode():
            self.flow.eval()
            theta, log_posterior = self.flow.sample(self.num_posterior_samples,
                                                    strain             = torch_strain,
                                                    restrict_to_bounds = False,
                                                    event_time         = event_time,   #restrict to bounds can be set to False given that if sample exceed prior bounds it is taken into account by prior weights
                                                    return_log_prob    = True,
                                                    verbose            = False)
        #compute medians
        medians = [float(torch.median(theta[par_name])) for par_name in self.inference_parameters]    

        #compute KS statistic if reference samples are provided
        if self.reference_posterior_samples:
            ks_stat = np.mean([(1-kstest(theta[name].cpu().numpy().T[0], self.reference_posterior_samples[name]).statistic)*100 for name in self.inference_parameters if not name == 'time_shift'])
        else:
            ks_stat = 'None'

        return theta, log_posterior, medians, ks_stat
   
    def compute_Bayes_factor(self, strain, whitened_strain, psd, event_time):
        r"""
        Method that computes the Bayes factor of signal vs noise hypotheses.
        The Bayes factor is defined as:

        .. math::

            \mathcal{B} = \frac{Z_{\text{model}}}{Z_{\text{noise}}}

        Taking the logarithm, we have:

        .. math::

            \log \mathcal{B} = \log Z_{\text{model}} - \log Z_{\text{noise}}
            
        Args:
        -----
            strain          (dict, TensorDict): Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            whitened_strain (dict, TensorDict): Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            psd             (dict, TensorDict): Dictionary containing interferometer Power Spectral Densities
            event_time                 (float): Strain (central) GPS time. It is used to correct the right ascension prediction
        
        Returns:
        --------
            logB      (float): Natural logarithm of the Bayes factor
            log10B    (float): Logarithm (base 10) of the Bayes factor
            IS_results (dict): Dictionary containing the results of the Importance Sampling method. See compute_model_evidence() method for details.
        """
        #run the Importance Sampling method    
        IS_results = self.compute_model_evidence(strain, whitened_strain, psd, event_time)
        
        #this is the model evidence
        logZ_model = IS_results['logZ']
        
        #get the noise evidence
        logZ_noise = self._noise_log_evidence(strain, psd)
        
        #compute the Bayes factors
        self._logB    = logZ_model - logZ_noise
        self._log10B  = self._logB/torch.log(torch.tensor(10))
        
        return self._logB, self._log10B, IS_results
    
    def compute_model_evidence(self, strain, whitened_strain, psd, event_time, normalize_weights=False):
        r"""
        Computes the model evidence \( Z \) using Importance Sampling.
        The model evidence is defined as:

        .. math::

            Z = \sum_{i} w_i \quad \text{where} \quad w_i = \frac{1}{N} \, \frac{p(\theta) \, L(s|\theta)}{q(\theta|s)}

        To mitigate numerical errors, the logarithm of the evidence is estimated instead.
        
        Args:
        -----
            strain          (dict, TensorDict): Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            whitened_strain (dict, TensorDict):  Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            psd             (dict, TensorDict):  Dictionary containing interferometer Power Spectral Densities
            event_time                 (float): Strain (central) GPS time. It is used to correct the right ascension prediction
            normalize_weights           (bool): If True, the importance weights are normalized. (Default False)
            
        Returns:
        --------
            IS_results (dict): Dictionary with the following keys 
                                - logZ         : Logarithm of the model evidence
                                - log_prior    : Logarithm of the prior probabilities
                                - logL         : Logarithm of the likelihoods
                                - log_posterior: Logarithm of the posterior probabilities
                                - log_weights  : Logarithm of the importance weights
                                - weights      : Importance weights
                                - eff          : Sampling efficiency
                                - valid_samples: Boolean mask of valid samples
                                - medians      : Medians of the posterior samples
                                - ks_stat      : Kolmogorov-Smirnov statistic comparing posterior samples to reference samples
        """
        #sampling flow posterior
        theta, log_posterior, medians, ks_stat  = self._sample_flow_posterior(whitened_strain, event_time)

        #add parameters not estimated by HYPERION
        if all(name in theta.keys() for name in ['M', 'q']) and not all(name in theta.keys() for name in ['m1', 'm2']):
            theta['m2'] = theta['q'] * theta['M'] / (1 + theta['q'])
            theta['m1'] = theta['M'] - theta['m2']
            theta.pop('M')
            theta.pop('q')
        
        elif all(name in theta.keys() for name in ['Mchirp', 'q']) and not all(name in theta.keys() for name in ['m1', 'm2']):
                theta['m1'] = (theta['Mchirp'] * (1 + theta['q'])**(1/5)) * theta['q']**(-3/5)
                theta['m2'] = theta['m1'] * theta['q']
                theta.pop('Mchirp')
                theta.pop('q')
      
        #compute log prior and valid samples
        log_prior = self.priors.log_prob(theta).double()
        valid_samples = log_prior!=-torch.inf
        
        #select valid-only posterior samples
        theta = theta[valid_samples].to('cpu')
        
        #compute log Likelihood
        logL = self.likelihood.log_Likelihood(strain=strain, psd=psd, theta=theta)

        #compute importance weights
        log_weights = log_prior[valid_samples] + torch.nan_to_num(logL) - log_posterior[valid_samples]
        weights     = torch.nan_to_num(torch.exp(log_weights - log_weights.max()))

        if normalize_weights:
            weights = (weights-weights.min())/(weights.max()-weights.min())
        
        #compute sampling efficiency
        eff = self.sample_efficiency(weights)
        
        #computing log Evidence
        N = torch.tensor(len(weights))
        logZ = torch.logsumexp(log_weights, 0) - torch.log(N)

        return dict(logZ          = logZ,
                    log_prior     = log_prior,
                    logL          = logL,
                    log_posterior = log_posterior,
                    log_weights   = log_weights,
                    weights       = weights,
                    eff           = eff,
                    valid_samples = valid_samples,
                    medians       = medians,
                    ks_stat       = ks_stat)