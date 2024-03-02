import h5py
import json
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import Dataset

from ...core.fft import *
from ...config import CONF_DIR
from ...core.distributions.prior_distributions import *
from ..asd import ASD_Sampler

from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwskysim.gwskysim.detectors import GWDetector
from gwskysim.gwskysim.utilities import pc_to_Msun

from tensordict import TensorDict


class GWDataset(Dataset):
    """
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    
    Given a specified prior and a waveform generator it generates as output a tuple
    (parameters, whitened_strain)

    """

    def __init__(self, 
                 dataset_filepath, 
                 detectors = ['L1', 'H1', 'V1'],
                 mode = 'training',
                 prior_filepath  = None, 
                 signal_duration = 1, 
                 noise_duration  = 8, 
                 device          = 'cpu',
                 inference_parameters = None,
                 extrinsic_parameters = None,
                 #random_seed     = 123,
                 whiten_kwargs   = None,
                 ):
        """
        Constructor.

        Args:
        -----
        waveform_generator: object
            GWskysim Waveform generator object istance. (Example: the EffectiveFlyByTemplate generator)

        asd_generators: dict of ASD_sampler objects
            Dictionary with hyperion's ASD_sampler object instances for each interferometer to simulate            

        prior_filepath: str or Path
            Path to a json file specifying the prior distributions over the simulation's parameters

        signal_duration: float
            Duration (seconds) of the output strain. (Default: 1)

        noise_duration : float
            Duration (seconds) of noise to simulate. Setting it higher than duration helps to avoid
            border discontinuity issues when performing whitening. (Default: 2)
        
        batch_size : int
            Batch size dimension. (Default: 512)
            
        device : str
            Device to be used to generate the dataset. Either 'cpu' or 'cuda:n'. (Default: 'cpu')

        inference_parameters : list of strings
            List of inference parameter names (e.g. ['m1', 'm2', 'ra', 'dec', ...]). (Default: 
            ['M', 'q', 'e0', 'p_0', 'distance', 'time_shift', 'polarization', 'inclination', 'ra', 'dec'])

        random_seed : int
            Random seed to set the random number generator for reproducibility. (Default: 123)
        
        """
        super(GWDataset, self).__init__()

        self.dataset_filepath     = dataset_filepath
        self.mode                 = mode
        self.signal_duration      = signal_duration
        self.noise_duration       = noise_duration
        self.device               = device
        self.inference_parameters = inference_parameters
        self.extrinsic_parameters = extrinsic_parameters

        self.whiten_kwargs = {'fftlength':4, 'overlap':2,
                              }
        if whiten_kwargs:
            self.whiten_kwargs.update(whiten_kwargs)

        #set up self random number generator
        #self.rng  = torch.Generator(device)
        #self.rng.manual_seed(random_seed)
        #self.seed = random_seed

        #load prior
        self._load_prior(prior_filepath)     

        #set up detectors and asd samplers
        self._setup_detectors(detectors)
        self._setup_asd_samplers(detectors)

        #open the dataset file
        self.hf = h5py.File(self.dataset_filepath, 'r')

        return
    
    def __len__(self):
        #list the keys in the hdf file
        samples = list(self.hf.keys())#.remove('parameters')
        return len(samples) -1
    
    @property
    def fs(self):
        return self._fs 
    @fs.setter
    def fs(self, value):
        self._fs = value
    
    @property
    def delta_t(self):
        return torch.tensor(1/self.fs)
    
    @property
    def det_names(self):
        return self.detectors.keys()

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
        return self._inference_parameters
    
    @inference_parameters.setter
    def inference_parameters(self, name_list):
        if name_list is None:
            name_list = ['M', 'q', 'e0', 'p_0', 'distance', 'time_shift', 'ra', 'dec'] #default
        self._inference_parameters = name_list

    @property
    def intrinsic_parameters(self):
        if not hasattr(self, '_intrinsic_parameters'):
            self._intrinsic_parameters = list(self.hf['parameters'].keys())           
        return self._intrinsic_parameters

    @property
    def extrinsic_parameters(self):
        return self._extrinsic_parameters
    
    @extrinsic_parameters.setter
    def extrinsic_parameters(self, name_list):
        if name_list is None:
            name_list = ['distance', 'time_shift', 'ra', 'dec']
        self._extrinsic_parameters = name_list

    @property
    def dataset_parameters(self):
        if not hasattr(self, '_dataset_parameters'):
            pars = dict()
            for p in self.intrinsic_parameters:
                pars[p] = torch.tensor(self.hf['parameters'][p])
            self._dataset_parameters = TensorDict.from_dict(pars)
        return self._dataset_parameters


    def _setup_detectors(self, detectors):
        """Sets up the GWDetector instances"""
        self.detectors = dict()
        for ifo in detectors:
            self.detectors[ifo] = GWDetector(ifo, 
                                             device=self.device, 
                                             reference_time=self.reference_time)

    def _setup_asd_samplers(self, detectors):
        """Sets up the ASD_Samplers instances"""
        self.asd_generator = dict()
        for i, ifo in enumerate(detectors):
            self.asd_generator[ifo] = ASD_Sampler(ifo, 
                                                  fs=self.fs, 
                                                  duration=self.noise_duration, 
                                                  random_seed=None)
            

    def _load_prior(self, prior_filepath=None):
        """
        Load the prior distributions specified in the json prior_filepath:
        if no filepath is given, the default one stored in the config dir will be used
        
        This function first reads the json file, then store the prior as a hyperion's MultivariatePrior instance. 
        Prior's metadata are stored as well. Metadata also contains the list of the inference parameters 
        The reference_time of the GWDetector instances is finally updated to the value set in the prior

        """
        #load the json file
        if not prior_filepath:
            print("[INFO]: No prior was given: loading default prior...")
            prior_filepath = CONF_DIR + '/BHBH-CE_population.json'

        with open(prior_filepath) as json_file:
            prior_kwargs = json.load(json_file)
            self._prior_metadata = prior_kwargs

        self.fs = prior_kwargs['fs']
        self.reference_time = prior_kwargs['reference_gps_time']
                    
        #load single priors as dictionary: each key is a parameter
        self.prior = dict()
        for i, p in enumerate(prior_kwargs['parameters'].keys()):
            dist = prior_kwargs['parameters'][p]['distribution']
            if dist == 'delta':
                val = prior_kwargs['parameters'][p]['value']
                self.prior[p] = prior_dict_[dist](val, self.device)
            else:
                min, max = prior_kwargs['parameters'][p]['min'], prior_kwargs['parameters'][p]['max']
                self.prior[p] = prior_dict_[dist](min, max, self.device)
        
        #convert prior dictionary to MultivariatePrior
        self.multivariate_prior = MultivariatePrior(self.prior)
        
        #add M and q to prior dictionary
        #NB: they are not added to MultivariatePrior to avoid conflict with the waveform_generator 
        #    this is intended when the inference parameters contain parameters that are combination of the default's one
        #    (Eg. the total mass M =m1+m2 or q=m2/m1 that have no simple joint distribution) 
        #    In this way we store however the metadata (eg. min and max values) without compromising the simulation 
             
        if ('M' in self.inference_parameters) and ('q' in self.inference_parameters):
            for p in ['M', 'q']:
                self.prior[p] = prior_dict_[p](self.prior['m1'], self.prior['m2'])
                min, max = float(self.prior[p].minimum), float(self.prior[p].maximum)
                metadata = {'distribution':p , 'min': min, 'max': max}
                self._prior_metadata['parameters'][p] = metadata
        
           
        #add inference parameters to metadata
        self._prior_metadata['inference_parameters'] = self.inference_parameters
        
        #add means and stds to metadata
        self._prior_metadata['means'] = self.means
        self._prior_metadata['stds']  = self.stds
        return 
    

    def _compute_M_and_q(self, prior_samples):

        #sorting m1 and m2 so that m2 <= m1
        m1 = prior_samples['m1']
        m2 = prior_samples['m2']
        m1, m2 = max(m1, m2), min(m1, m2)
        
        prior_samples['M'] = m1+m2
        prior_samples['q'] = m2/m1
        
        return prior_samples
    
    
    def standardize_parameters(self, prior_samples):
        """Standardize prior samples to zero mean and unit variance"""
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            standardized = self.prior[parameter].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized)
        out_prior_samples = torch.tensor(out_prior_samples)
        return out_prior_samples
    

    def sample_extrinsic(self):
        samples = dict()
        for p in self.extrinsic_parameters:
            samples[p] = self.prior[p].sample(1)
        return samples

    def _read_hdf(self, idx):
        """Read the specified idx waveforms from the hdf file"""
        hp, hc = np.zeros((2, self.signal_duration*self.fs))
        self.hf[str(idx)]['hp'].read_direct(hp)
        self.hf[str(idx)]['hc'].read_direct(hc)
        return hp, hc
    
    def _project_wave(self, hp, hc, ra, dec, polarization, time_shift):
        #window = torch.hann_window(self.signal_duration*self.fs)
        pad_len = (self.noise_duration-self.signal_duration)*self.fs //2
        h = dict()
        for ifo in self.det_names:
            dt = self.detectors[ifo].time_delay_from_earth_center(ra, dec)+time_shift
            projected = self.detectors[ifo].project_wave(hp, hc, ra, dec, polarization)
            projected = np.pad(projected, (pad_len, pad_len))
            projected = torch.from_numpy(projected)
            projected_f = rfft(projected, n = projected.shape[-1], 
                               fs=self.fs)*torch.exp(-1j*2*torch.pi*self.frequencies*dt)
            h[ifo] = irfft(projected_f, fs=self.fs).numpy()
        return h
    
    @staticmethod
    def _rescale_waveforms_to_distance(hp, hc, old_distance, new_distance):
        """Rescales plus&cross gw polarizations to a new luminosity distance"""
        #convert distances to solar masses
        d_old = pc_to_Msun(old_distance*1e6)
        d_new = pc_to_Msun(new_distance*1e6)
        
        hp_new = hp*d_old/d_new
        hc_new = hc*d_old/d_new
        return hp_new, hc_new
    
    def _add_noise_and_whiten(self, h):
        noise_points = self.noise_duration*self.fs
        middle = noise_points//2
        tcrop = self.signal_duration*self.fs//2
        
        whitened_strain = dict()
        for ifo in self.det_names:
            #adding noise
            _, noise = self.asd_generator[ifo].sample(batch_size=1, noise_points=noise_points)
            
            strain = h[ifo] + noise.squeeze(0).numpy()
            
            #whitening
            strain = TimeSeries(strain, t0=-self.noise_duration/2, dt=1/self.fs)
            ws = np.array(strain.whiten(**self.whiten_kwargs))
            
            #cropping to desired duration
            whitened_strain[ifo] = ws[middle-tcrop:middle+tcrop]
        
        return whitened_strain

    def __getitem__(self, idx=None):
        
        #read waveform polarizations and associated parameters
        hp, hc = self._read_hdf(idx)
        parameters = self.dataset_parameters[idx]
        parameters = self._compute_M_and_q(parameters)
        old_distance = parameters['distance'].item()
    
        #sample extrinsic parameters
        parameters.update(self.sample_extrinsic())
        
        #rescale waveforms to new luminosity distance
        hp, hc = self._rescale_waveforms_to_distance(hp, hc, 
                                                     old_distance, 
                                                     parameters['distance'].item())
        
        h = self._project_wave(hp, hc, 
                               parameters['ra'].item(), 
                               parameters['dec'].item(), 
                               parameters['polarization'].item(),
                               parameters['time_shift'].item()
                               )
        
        whitened_strain = self._add_noise_and_whiten(h)
        
        torch_strain = torch.zeros((len(self.det_names), self.signal_duration*self.fs))
        for i, ifo in enumerate(self.det_names):
            torch_strain[i] = torch.from_numpy(whitened_strain[ifo])

        parameters = self.standardize_parameters(parameters)
        
        return parameters, torch_strain
