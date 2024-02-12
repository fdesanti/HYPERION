import torch
from torch.utils.data import Dataset
from torch.distributions import Normal 

from ..core.fft import *
from ..core.distributions.prior_distributions import *
from ..config import CONF_DIR

import json

class DatasetGenerator(Dataset):
    """
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    
    Given a specified prior and a waveform generator it generates as output a tuple
    (parameters, whitened_strain)

    """

    def __init__(self, 
                 waveform_generator, 
                 asd_generators, 
                 prior          = None, 
                 duration       = 1, 
                 noise_duration = 2, 
                 batch_size     = 512,
                 device         = 'cpu'
                 ):
        """
        Constructor.

        Args:
        -----
        waveform_generator: object
            GWskysim Waveform generator object istance. (Example: the EffectiveFlyByTemplate generator)

        asd_generators: dict of ASD_sampler objects
            Dictionary with hyperion's ASD_sampler object instances for each interferometer to simulate            

        prior: object 
            hyperion's MultivariatePrior object instance defining the prior on the population parameters

        duration: float
            Duration (seconds) of the output strain. (Default: 1)

        noise_duration : float
            Duration (seconds) of noise to simulate. Setting it higher than duration helps to avoid
            border discontinuity issues when performing whitening. (Default: 2)
        
        batch_size : int
            Batch size dimension. (Default: 512)
            
        device : str
            Device to be used to generate the dataset. Either 'cpu' or 'cuda:n'. (Default: 'cpu')
        
        """
        super(DatasetGenerator, self).__init__()

        self.waveform_generator  = waveform_generator
        self.asd_generator       = asd_generators
        self.batch_size          = batch_size
        self.duration            = duration
        self.noise_duration      = noise_duration
        self.device              = device

        #self.window_len          = self.fs//16

        assert sorted(waveform_generator.det_names) == sorted(asd_generators.keys()), f"Mismatch between ifos in waveform generator\
                                                                                       and asd_generator. Got {sorted(waveform_generator.det_names)}\
                                                                                       and {sorted(asd_generators.keys())}, respectively "
        
        if prior is None:
            print("---> No prior was given: loading default prior...")
            self._load_default_prior()     
        else:
            self.prior = prior
        
        return

    @property
    def fs(self):
        return self.waveform_generator.fs
    
    @property
    def delta_t(self):
        return torch.tensor(1/self.fs)
    
    @property
    def det_names(self):
        return self.waveform_generator.det_names

    @property
    def frequencies(self):
        if not hasattr(self, '_frequencies'):
            n = (self.noise_duration) * self.fs
            self._frequencies = rfftfreq(n, d=1/self.fs)
        return self._frequencies
    
    @property
    def noise_std(self):
        return 1 / torch.sqrt(2*self.delta_t)
    

    @property
    def means(self):
        if not hasattr(self, '_means'):
            self._means = dict()
            for parameter in self.inference_parameters:
                self._means[parameter] = float(self.prior[parameter].mean)
        return self._means
    
    @property
    def stds(self):
        if not hasattr(self, '_stds'):
            self._stds = dict()
            for parameter in self.inference_parameters:
                self._stds[parameter] = float(self.prior[parameter].std)
        return self._stds

    @property
    def inference_parameters(self):
        if not hasattr(self, '_inference_parameters'):
            self._inference_parameters = ['M', 'q', 'e0', 'p_0', 'distance', 'time_shift', 'polarization', 'inclination', 'ra', 'dec'] #default
        return self._inference_parameters
    
    @inference_parameters.setter
    def inference_parameters(self, name_list):
        self._inference_parameters = name_list


    def _load_default_prior(self):
        self.prior = dict()
        
        prior_file = CONF_DIR + '/BHBH-CE_population.json'
        with open(prior_file) as json_file:
            prior_kwargs = json.load(json_file)['parameters']
            self._prior_metadata = prior_kwargs
            
        for parameter in prior_kwargs.keys():
            dist = prior_kwargs[parameter]['distribution']
            if dist == 'delta':
                val = prior_kwargs[parameter]['value']
                self.prior[parameter] = prior_dict_[dist](val, self.device)
            else:
                min, max = prior_kwargs[parameter]['min'], prior_kwargs[parameter]['max']
                self.prior[parameter] = prior_dict_[dist](min, max, self.device)
        
        
        self.multivariate_prior = MultivariatePrior(self.prior)
        
        #add M and q to prior        
        if ('M' in self.inference_parameters) and ('q' in self.inference_parameters):
            for p in ['M', 'q']:
                self.prior[p] = prior_dict_[p](self.prior['m1'], self.prior['m2'])
                min, max = float(self.prior[p].minimum), float(self.prior[p].maximum)
                metadata = {'distribution':p , 'min': min, 'max': max}
                self._prior_metadata[p] = metadata
            
        return 
    
    def __len__(self):
        return int(1e8) #set it very high. It matters only when torch DataLoaders are used
    
    
    def _compute_M_and_q(self, prior_samples):

        #sorting m1 and m2 so that m2 <= m1
        m1 = prior_samples['m1'].T.squeeze()
        m2 = prior_samples['m2'].T.squeeze()

        m, _ = torch.sort(torch.stack([m1, m2]).T)
        
        m1 = m.T[1]
        m2 = m.T[0]
        
        #m1 and m2 have shape [Nbatch]
        prior_samples['M'] = (m1+m2).reshape((self.batch_size, 1))
        prior_samples['q'] = (m2/m1).reshape((self.batch_size, 1))

        return prior_samples


    def _apply_time_shifts_and_whiten(self, h):

        out_h = []
        pad   = (self.noise_duration - self.duration)*self.fs // 2
        
        for ifo in self.det_names:
                        
            dt = h[f"{ifo}_delay"] #time delay from Earth center + central time shift 
            
            #apply hann window to cancel border offsets
            h[ifo]*= torch.hann_window(h[ifo].shape[-1])
        
            #pad template with left/right last values adding points up to noise_duration
            h_tmp  = torch.nn.functional.pad(h[ifo], (pad, pad ), mode='replicate')  

            #apply time shifts to templates in the frequency domain
            h_tmp = rfft(h_tmp, n = h_tmp.shape[-1], fs=self.fs) * torch.exp(-1j * 2 * torch.pi * self.frequencies * dt)

            #sample asd from generator
            asd = self.asd_generator[ifo](batch_size = self.batch_size)
            
            #divide frequency domain template with the ASD
            h_tmp /= asd
            
            #revert back to time domain
            h_tmp = irfft(h_tmp, fs = self.fs)
    
            #crop to desired output duration
            central_time = h_tmp.shape[-1]//2
            d = self.duration * self.fs // 2 
            out_h.append(h_tmp[:, central_time-d : central_time+d])

        out_h = torch.stack(out_h, dim=1)
        
        return out_h / self.noise_std
    
    
    def _add_noise(self, h):
        noise = Normal(loc = 0, scale = self.noise_std).sample(h.shape)
        #noise = torch.randn(h.shape)
        return (h + noise)/self.noise_std
    
    
    def standardize_parameters(self, prior_samples):
        """Standardize prior samples to zero mean and unit variance"""
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            standardized = self.prior[parameter].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized)
            if parameter == 'dec':
                print(standardized)
            
        out_prior_samples = torch.cat(out_prior_samples, dim=-1)
        return out_prior_samples
    


    def __getitem__(self, idx=None):

        #sampling prior
        prior_samples = self.multivariate_prior.sample((self.batch_size, 1))
        #prior_samples['t0_p'] = prior_samples['t0_p'][0]

        out_prior_samples = self._compute_M_and_q(prior_samples.copy())
        
        #generate projected waveform strain
        h = self.waveform_generator(**prior_samples)

        #apply time shift and whiten
        whitened_template = self._apply_time_shifts_and_whiten(h)
        
        #add gaussian noise
        whitened_strain = self._add_noise(whitened_template)

        #standardize parameters
        out_prior_samples = self.standardize_parameters(out_prior_samples)
    
    
        return out_prior_samples, whitened_strain
    

