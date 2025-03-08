import os
import yaml
import torch

from tensordict import TensorDict

from ..core import HYPERION_Logger
from ..core.fft import *
from ..simulations import WhitenNet
from ..core.distributions import MultivariatePrior, prior_dict_
from ..core.distributions import q_uniform_in_components as q_prior

log = HYPERION_Logger()

class DatasetGenerator:
    r"""
    Class to generate training dataset. Can work either offline as well as an online (i.e. on training) generator.
    The class is initialized with a waveform generator, a set of ASD generators, a detector network and a prior file.
    
    The dataset is simulated in the following steps:

        1. Sample intrinsic parameters from the intrinsic prior 

        2. Generate the corresponding waveforms

        3. Sample extrinsic parameters from the extrinsic prior

        4. Project the waveforms onto the detectors

        5. Compute the relative time shifts

        6. Sample the ASD and whiten the strain

        7. Standardize the parameters

        8. Return the standardized parameters and the whitened strain as a tuple

    Args:
        waveform_generator (WaveformGenerator): The WaveformGenerator object.
        asd_generators   (dict of ASD_Sampler): A dictionary containing the ASD_Sampler generators for each detector.
        det_network        (GWDetectorNetwork): The GWDetectorNetwork instance.
        prior_filepath                  (Path): The path to the prior file.
        batch_size                       (int): The batch size. (Default: 512)
        device                           (str): The device to use ('cpu' or 'cuda'). (Default: 'cpu')
        random_seed                      (int): The random seed to be passed to the MultivariatePrior instances for reproducibility. (Default: None)
        num_preload                      (int): The number of waveforms to preload before each epoch. (Default: 1000)
        n_proc                           (int): The number of parallel processes to use for the waveform generation. (Default: ``2 * os.cpu_count() // 3``)
        use_reference_asd               (bool): Whether to use the reference ASD for whitening. (Default: False)
        inference_parameters     (list of str): The list of parameters to infer. (Default: None)
        whiten_kwargs                         : The kwargs to be passed to the WhitenNet instance. (Default: None)

    Hint:
        The ``inference_parameters`` list allows to specify the parameters to be inferred different to those in the prior. 
        It might be just a subset or a combination of the prior parameters.
        For instance if the prior contains the parameters :math:`(m_1, m_2)` then one can infer the total mass :math:`M = m_1 + m_2`, 
        the chirp mass :math:`\mathcal{M} = \dfrac{(m_1m_2)^{3/5}}{M^{1/5}}` or the mass ratio :math:`q = m_2/m_1`.

    Note: 
        By default the class uses the GWPy's whitening method. See the WhitenNet class documentation for further details.
    """
    def __init__(self, 
                 waveform_generator, 
                 asd_generators, 
                 det_network,
                 prior_filepath       = None,
                 batch_size           = 512,
                 device               = 'cpu',
                 random_seed          = None,
                 num_preload          = 1000,
                 n_proc               = 2 * os.cpu_count() // 3,
                 use_reference_asd    = False,
                 inference_parameters = None,
                 **whiten_kwargs,
                 ):
    
        
        self.waveform_generator   = waveform_generator
        self.asd_generator        = asd_generators
        self.batch_size           = batch_size
        self.device               = device
        self.inference_parameters = inference_parameters
        self.det_network          = det_network
        self.n_proc               = n_proc
        self.use_reference_asd    = use_reference_asd
        #self.whiten_kwargs        = whiten_kwargs
        #print(self.whiten_kwargs)
        self.whitening_method     = whiten_kwargs.get('method', 'gwpy')
        self.whitening_normalize  = whiten_kwargs.get('normalize', True)

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

        self.WhitenNet = WhitenNet(duration = waveform_generator.duration, 
                                   fs       = waveform_generator.fs,
                                   device   = device,
                                   rng      = self.rng)
        

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
        
        This function first reads the yml file, then store the prior as a hyperion's MultivariatePrior instance. 
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

        #NOTE - this requires m1 and m2 to be uniformly sampled in the prior
        for p in ['M', 'Mchirp', 'q']:
            #NOTE - the following if statement is skipped if the parameter is already in the prior
            #       (e.g. if we sample directly in total mass M and mass ratio q with certain priors)
            if p in self.inference_parameters and not p in self.full_prior.keys():
                #NOTE - this second if statement is needed specifically when Mchirp is in the inference parameters
                #       but not in the prior. Therefore we must check if we are sampling m1/m2 or M/q
                #       because the Mchirp_uniform_in_components prior accepts both options
                if 'm1' in self.full_prior.keys() and 'm2' in self.full_prior.keys():
                    kwargs = {'m1': self.full_prior['m1'], 'm2': self.full_prior['m2']}
                elif 'M' in self.full_prior.keys() and 'q' in self.full_prior.keys():
                    kwargs = {'M': self.full_prior['M'], 'q': self.full_prior['q']}
                else:
                    raise ValueError('Cannot compute M, Mchirp and q from the prior samples')
                
                self.full_prior[p] = prior_dict_[p](**kwargs)
                min, max = float(self.full_prior[p].minimum), float(self.full_prior[p].maximum)
                
                metadata = {'distribution':f'{p}_uniform_in_components', 'kwargs':{'minimum': min, 'maximum': max}}
                self.prior_metadata['parameters'][p] = metadata

        #store means and stds
        self.prior_metadata['means'] = self.means
        self.prior_metadata['stds']  = self.stds
    

    def _compute_M_Mchirp_and_q(self, prior_samples):
        
        #check wether m1 and m2 are sampled by the prior
        if all([p in prior_samples.keys() for p in ['m1', 'm2']]):

            #sorting m1 and m2 so that m2 <= m1
            m1, m2 = prior_samples['m1'], prior_samples['m2']
            m1, m2 = q_prior._sort_masses(m1, m2)
            
            #m1 and m2 have shape [Nbatch]
            if 'Mchirp' in self.inference_parameters:
                prior_samples['Mchirp'] = (m1*m2)**(3/5)/(m1+m2)**(1/5)
            
            if 'M' in self.inference_parameters:
                prior_samples['M'] = (m1+m2)
            
            if 'q' in self.inference_parameters:
                prior_samples['q'] = (m2/m1)
        
        #otherwise we check if Mchirp and q are sampled by the prior
        elif all([p in prior_samples.keys() for p in ['Mchirp', 'q']]):
            
            Mchirp, q = prior_samples['Mchirp'], prior_samples['q']
            m1 = Mchirp*(1+q)**(1/5)/q**(3/5)
            m2 = q*m1

            if 'M' in self.inference_parameters:
                prior_samples['M'] = (m1+m2)
            
            if 'q' in self.inference_parameters:
                prior_samples['q'] = (m2/m1)

        #otherwise we check if M and q are sampled by the prior
        elif all([p in prior_samples.keys() for p in ['M', 'q']]):
            
            M, q = prior_samples['M'], prior_samples['q']
            m1 = M/(1+q)
            m2 = q*m1
            
            if 'Mchirp' in self.inference_parameters:
                prior_samples['Mchirp'] = (m1*m2)**(3/5)/(m1+m2)**(1/5)
        
        #any other combination is not allowed
        else:
            raise ValueError('Cannot compute M, Mchirp and q from the prior samples')

        return prior_samples
         

    def standardize_parameters(self, prior_samples):
        """
        Standardize prior samples to zero mean and unit variance
        """
        
        out_prior_samples = []
        for parameter in self.inference_parameters:
            standardized = self.full_prior[parameter].standardize_samples(prior_samples[parameter])
            out_prior_samples.append(standardized)
            
        out_prior_samples = torch.cat(out_prior_samples, dim=-1)
        return out_prior_samples
    
    
    def get_idxs(self):
        """
        Return a set of batch indices to sample the preloaded waveforms
        """
        if not hasattr(self, 'preloaded_wvfs'):
            raise ValueError('There are no preloaded waveforms. Please run pre_load_waveforms() first.')

        idxs = torch.arange(self.num_preload).float()
        return torch.multinomial(idxs, self.batch_size, replacement=False)
    

    def preload_waveforms(self):
        """
        Preload a set of waveforms. This function must be called at the beginning of each epoch.
        """
        
        log.info('Preloading a new set of waveforms...')

        #first we sample the intrinsic parameters
        self.prior_samples = self.intrinsic_prior.sample(self.num_preload)
        
        #compute M, Mchirp and/or q and update prior samples
        self.prior_samples = self._compute_M_Mchirp_and_q(self.prior_samples)
        
        #then we call the waveform generator
        hp, hc, tcoal = self.waveform_generator(self.prior_samples.to('cpu'), 
                                                n_proc=self.n_proc)
        log.info('Done.')

        #store the waveforms as a TensorDict
        wvfs = {'hp': hp, 'hc': hc}
        tcoals = {'tcoal': tcoal}
        
        self.preloaded_wvfs = TensorDict.from_dict(wvfs).to(self.device)
        self.tcoals = TensorDict.from_dict(tcoals).to(self.device)
        
        return


    def __getitem__(self, add_noise=True):
        """
        Generate a batch of whitened strain and corresponding parameters.

        Args:
            add_noise (bool): Whether to add noise to the whitened strain. (Default: True)

        Returns:
            tuple: A tuple containing the standardized parameters, the whitened strain and the ASD.
        """

        idxs = self.get_idxs()

        #get the prior samples
        prior_samples = self.prior_samples[idxs].unsqueeze(1)

        #get the corresponding preloaded waveforms
        hp, hc = self.preloaded_wvfs[idxs]['hp'], self.preloaded_wvfs[idxs]['hc']

        #sample extrinsic priors
        prior_samples.update(self.extrinsic_prior.sample((self.batch_size, 1)))

        #rescale luminosity distance
        hp /= prior_samples['luminosity_distance']
        hc /= prior_samples['luminosity_distance']
                         
        #project strain onto detectors
        h = self.det_network.project_wave(hp, hc, 
                                          ra           = prior_samples['ra'],
                                          dec          = prior_samples['dec'],
                                          polarization = prior_samples['polarization'])
        #compute relative time shifts
        time_shifts = self.det_network.time_delay_from_earth_center(ra  = prior_samples['ra'], 
                                                                    dec = prior_samples['dec'])        
        
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(h['L1'][0].cpu().numpy())
        #plt.show()
        for det in h.keys():
            time_shifts[det] += prior_samples['time_shift']
        

        #sample asd --> whiten --> add noise
        #asd = {det: self.asd_generator[det].sample(self.batch_size) for det in self.det_network.detectors}
        asd = {}
        noise = {}
        for det in self.det_network.detectors:
            #asd[det], noise[det] = self.asd_generator[det].sample(self.batch_size, noise=True)
            #asd[det] = self.asd_generator[det].asd_reference
            asd[det] = self.asd_generator[det].sample(self.batch_size, 
                                                      noise = False, 
                                                      use_reference_asd=self.use_reference_asd)
        torch_asd = torch.stack([asd[det] for det in self.det_network.detectors], dim=1)
        
        whitened_strain = self.WhitenNet(h=h, 
                                         asd        = asd,
                                         noise      = None,
                                         time_shift = time_shifts,
                                         add_noise  = add_noise,
                                         method     = self.whitening_method,
                                         normalize  = self.whitening_normalize)

        #standardize parameters
        prior_samples['tcoal'] = self.tcoals[idxs]['tcoal'] + prior_samples['time_shift']#add tcoal to time_shift
        out_prior_samples = self.standardize_parameters(prior_samples)

        #convert to a single float tensor
        out_whitened_strain = torch.stack([whitened_strain[det] for det in self.det_network.detectors], dim=1)
        return out_prior_samples.float(), out_whitened_strain.float(), torch_asd.float()