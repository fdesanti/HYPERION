import torch 
import numpy as np

from tensordict import TensorDict
from scipy.stats import kstest
from .likelihood import  GWLikelihood
from ...core.distributions.prior_distributions import * 


class IS_Priors(MultivariatePrior):
    """ 
        Wrapper to the MultivariatePrior class.
        
        Args:
        ----- 
            flow : nn.Module
                flow model. Its prior is used to initialized the various parameter priors
                
            device : str 
                either 'cpu' or 'cuda' to enable GPU.  (Default 'cpu')       
    """

    def __init__(self, flow, device = 'cpu'):
        
        self.flow = flow        
        
        priors = flow.prior_metadata['parameters'].copy()

        for p in ['M', 'Mchirp', 'q']:
            if p in priors:
                priors.pop(p)

        super(IS_Priors, self).__init__(priors, device=device)
        return
    


class ImportanceSampling():    
    
    """Class that performs Importance Sampling to estimate the Bayes Factor = Z_model/Z_noise
    
    Z_model = 1/N sum [p(theta) p (s|theta)]/q(theta|s)
    
    Z_noise = noise likelihood

    """
    def __init__(self, 
                     flow, 
                     waveform_generator, 
                     device = 'cpu', 
                     num_posterior_samples = 1000, 
                     reference_posterior_samples = None):
        
        """
        Importance Sampling initialization

        Args: 
        -----
            flow : 
                Trained Normalizing Flow model
           
            device : str 
                either 'cpu' or 'cuda' to enable GPU.  (Default 'cpu')
           
            num_posterior_samples : int
                number of posterior samples to draw from the flow
                
            waveform_generator : 
                GW Waveform generator object
                
        """
        
        if flow is not None:
            self.flow   = flow.to(device)
            self.priors = IS_Priors(flow, device)
            
        self.device = device
        self.num_posterior_samples = num_posterior_samples
        self.waveform_generator    = waveform_generator
        
        self.reference_posterior_samples = reference_posterior_samples

        self.likelihood = GWLikelihood(waveform_generator, device=device)
        
        return

    
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
   
    def compute_Bayes_factor(self, strain, whitened_strain, psd, event_time):
        """
            Method that computes the Bayes factor of signal vs noise hypotheses
            
                    B = Z_model/Z_noise  -> logB = logZ_model - logZ_noise
                
            Args:
            -----
                strain : dict
                    Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            
                whitened_strain : dict
                    Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
                    
                psd : dict
                    Dictionary containing interferometer Power Spectral Densities
                    
                event_time : float
                    strain (central) GPS time. It is used to correct the right ascension prediction
         
            Returns:
            --------
                logB : float
                    Natural logarithm of the Bayes factor
                
                log10B : float
                    Logarithm (base 10) of the Bayes factor
        """
                
        is_results = self.compute_model_evidence(strain, whitened_strain, psd, event_time)
        logZ_model = is_results['logZ']
        sample_efficiency = is_results['eff']
        parameters_medians = is_results['medians']
        ks_stat = is_results['ks_stat']
        
        logZ_noise = self._noise_log_evidence(strain, psd)
        
        self._logB    = logZ_model - logZ_noise
        self._log10B  = self._logB/torch.log(torch.tensor(10))
        
        #print(f'logZ model {logZ_model} - logZ noise {logZ_noise} - logB {self._logB} - log10B {self._log10B} - eff {sample_efficiency}')
        
        return self._logB, self._log10B, sample_efficiency, parameters_medians, ks_stat
    
    
    def compute_model_evidence(self, strain, whitened_strain, psd, event_time):
        """
            Computes the model evidence Z with Importance Sampling
            
                Z = sum w_i , where w_i = (1/N) * [p(\theta)*L(s|\theta)]/q(\theta|s) 
            
            To reduce numerical errors it is estimated log Z
            
            
            Args:
            -----
                strain : dict
                    Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
            
                whitened_strain : dict
                    Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
                    
                psd : dict
                    Dictionary containing interferometer Power Spectral Densities
                    
                event_time : float
                    strain (central) GPS time. It is used to correct the right ascension prediction
                    
            Returns:
            --------
                thetas : dict or pandas.DataFrame
                    Posterior samples
                    
                log_posterior : torch.tensor
                    log flow posterior defined as log q(\theta| s) = log N(f^-1(u)) + log J(f^-1(u))
        """
        
        
        #sampling flow posterior
        thetas, log_posterior, medians, ks_stat  = self._sample_flow_posterior(whitened_strain, event_time)
        
        '''
        import corner
        import matplotlib.pyplot as plt
        p = thetas.copy()
        for name in p.keys():
            p[name] = p[name].squeeze().cpu().numpy()
        corner.corner(p)
        plt.show()
        '''
        
        #add parameters not estimated by HYPERION

        if 'M' in thetas.keys() and 'q' in thetas.keys():
        
            thetas['m2'] = thetas['q'] * thetas['M'] / (1 + thetas['q'])  #m2 = qM/(1+q)   m1 = M-m2
            thetas['m1'] = thetas['M'] - thetas['m2']
            thetas.pop('M')
            thetas.pop('q')
            if 'Mchirp' in thetas.keys():
                thetas.pop('Mchirp')
        
        elif 'Mchirp' in thetas.keys() and 'q' in thetas.keys():
            thetas['m1'] = (thetas['Mchirp'] * (1 + thetas['q'])**(1/5)) * thetas['q']**(-3/5)
            thetas['m2'] = thetas['m1'] * thetas['q']
            thetas.pop('Mchirp')
            thetas.pop('q')
        
        
        #FIXME - in the case of TEOBResumS j_hyp might be missing/it's better to manage
        #        priors with the MultiVariate Prior class
        if not 'j_hyp' in thetas.keys():
            thetas['j_hyp'] = torch.tensor([4.0]*len(thetas['m1'])).to(self.device)
        
        #compute log Prior
        '''
        #sample parameters not estimated by HYPERION
        unsampled_parameters = self.priors.names - thetas.keys()
        for name in unsampled_parameters:
            thetas[name] = self.priors.priors[name].sample([self.num_posterior_samples])
        '''
        
        #compute log prior and valid samples
        logP = self.priors.log_prob(thetas).double()
        valid_samples = logP!=-torch.inf
        
        #select valid-only posterior samples
        thetas = thetas[valid_samples].to('cpu')
        
    
        #compute log Likelihood
        #logL = torch.empty_like(logP)
        logL = self.likelihood.log_Likelihood(strain=strain, psd=psd, waveform_parameters=thetas)
        #logL[valid] = logL_valid
        
        

        log_weights =  logP[valid_samples] + torch.nan_to_num(logL) - log_posterior[valid_samples]
        
        weights = torch.nan_to_num(torch.exp(log_weights.max()/log_weights))
        #wei

        #weights = torch.exp(log_weights - torch.logsumexp(log_weights, 0))
        #weights = torch.exp(log_weights - torch.max(log_weights))

        weights = (weights-weights.min())/(weights.max()-weights.min())
        
        #print(f'log weights {log_weights} / weights {weights}')
        eff = self.sample_efficiency(weights)
        #print(f'SAMPLE EFFICIENCY {eff*100:.3f}%\n Tot samples {len(weights)}\n')
        
        

        N = torch.tensor(len(weights))
        
        #computing log Evidence
        logZ = torch.logsumexp(log_weights, 0) - torch.log(N)
        
        #print(f"logP {logP} \n logP nan_num {torch.nan_to_num(logP)} \n logL {logL} \n weights {log_weights} \n logZ {logZ} \n log_posterior {log_posterior}")

        stats={'logP': logP, 'logL': logL, 'log_posterior': log_posterior, 'log_weights': log_weights, 'weights': weights, 'valid_samples':valid_samples}

        return {'logZ': logZ, 'eff': eff, 'medians': medians, 'ks_stat': ks_stat, 'stats': stats}
    
    
    def sample_efficiency(self, weights):
        N = len(weights)
        #eff = (1/N)*torch.sum(weights)**2/torch.sum(weights**2)
        eff = (1/N)* (weights.sum())**2 / (weights**2).sum()
        return eff*100
        


    def _noise_log_evidence(self, strain, psd):
        """
            Computes log Z assuming the null hypothesis of having noise only data
            
                Z = L(s | \theta)
            
            Args:
            -----
                strain : dict
                    Raw strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
                    
                psd : dict
                    Dictionary containing interferometer Power Spectral Densities
            
            Returns:
            --------
                logL_noise : torch.tensor
                    Log noise Likelihood
        """ 
        return self.likelihood.noise_log_Likelihood(strain, psd)
    
    
    def _sample_flow_posterior(self, whitened_strain, event_time):
        """
            Samples the posterior for the given stretch of data
            
            Args:
            -----
                whitened_strain : dict
                    Whitened strain time series tensor of shape [1, Num detectors, 1s * sampling_frequency]
                    
                event_time : float
                    strain (central) GPS time. It is used to correct the right ascension prediction
                    
            Returns:
            --------
                thetas : dict or pandas.DataFrame
                    Posterior samples
                    
                log_posterior : torch.tensor
                    log flow posterior defined as log q(\theta| s) = log N(f^-1(u)) + log J(f^-1(u))
            
        """
        torch_strain = torch.stack([whitened_strain[det] for det in self.det_names]).unsqueeze(0)
        #torch_strain = torch.from_numpy(torch_strain).float().unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            self.flow.eval()
            thetas, log_posterior = self.flow.sample(self.num_posterior_samples, strain  = torch_strain, 
                                                    restrict_to_bounds=False, event_time = event_time, #restrict to bounds can be set to False given that if sample exceed prior bounds it is taken into account by prior weights
                                                    return_log_prob=True,
                                                    verbose=False) 
        
        medians = [float(torch.median(thetas[par_name])) for par_name in self.inference_parameters]    



        if self.reference_posterior_samples:
            ks_stat = np.mean([(1-kstest(thetas[name].cpu().numpy().T[0], self.reference_posterior_samples[name]).statistic)*100 for name in self.inference_parameters if not name == 'time_shift'])
        else:
            ks_stat = 'None'
        return thetas, log_posterior, medians, ks_stat
    
    
    
    
    
    
    
    
    

        

        
        
