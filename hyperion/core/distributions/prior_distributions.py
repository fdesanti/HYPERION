import torch
from tensordict import TensorDict
from torch.distributions import Gamma

N_ = int(1e7)

#=======================================
# Rand Wrapper Class
#=======================================
class Rand():
    """
    Wrapper class to torch.rand to explot a random generator with a fixed seed
    for reproducibility
    """
    def __init__(self, seed = None, device = 'cpu'):
        
        if seed is not None:
            self.rng = torch.Generator(device)
            self.rng.manual_seed(seed)
        else:
            self.rng = None
        

    def __call__(self, sample_shape, device, dtype=None):
        return torch.rand(sample_shape, generator = self.rng, device = device, dtype=dtype)



#=======================================
# Analytical Prior Distributions
#=======================================
class BasePrior():
    """Base class for Prior Distributions"""
    
    def __init__(self, minimum=None, maximum=None, device='cpu', seed=None):

        
        self.device  = device
        self.rand    = Rand(seed, device)
        self.minimum = minimum
        self.maximum = maximum
        assert self.maximum >= self.minimum, f"maximum must be >= than minimum, given {maximum} and {minimum} respectively"

        return
    
    @property
    def minimum(self):
        return self._minimum
    @minimum.setter
    def minimum(self, value):
        if value is not None:
            self._minimum = torch.tensor(value).to(self.device)
        else:
            self._minimum = self.sample((N_)).min()
    
    @property
    def maximum(self):
        return self._maximum
    @maximum.setter
    def maximum(self, value):
        if value is not None:
            self._maximum = torch.tensor(value).to(self.device)
        else:
            self._maximum = self.sample((N_)).max()
        
        
    @property
    def mean(self):
        #print('default')
        if not hasattr(self, '_mean'):
            self._mean = self.sample((N_)).mean()
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = self.sample((N_)).std()
        return self._std
    
    def is_in_prior_range(self, samples):
        """Returns 1 if samples is in the prior boundaries, 0 otherwise
        """
        return ( (samples >= self.minimum) & (samples <= self.maximum) ).bool().int()
    
    def standardize_samples(self, samples):
        return (samples - self.mean) / self.std
    
    def de_standardize_samples(self, samples):
        return (samples * self.std) + self.mean
    
    def log_prob(self, samples):
        raise NotImplementedError
    
    def rescale(self, sample_shape):
        """Rescale U(0, 1) samples to the desired range and prior"""
        raise NotImplementedError
    
    def sample(self, sample_shape):
        raise NotImplementedError
    
    


class UniformPrior(BasePrior):

    def __init__(self, minimum, maximum, device = 'cpu', seed = None):
        super(UniformPrior, self).__init__(minimum, maximum, device, seed)
        return
    
    @property
    def name(self):
        return 'uniform'
    
    @property
    def mean(self):
        if not hasattr(self, '_mean'):
            self._mean = (self.minimum + self.maximum) / 2
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = (1/12)**0.5 * (self.maximum - self.minimum)
        return self._std
    
    def log_prob(self, samples):
        return torch.xlogy(1, (samples >= self.minimum) & (samples <= self.maximum)) - torch.xlogy(1, self.maximum - self.minimum)

    def rescale(self, samples):
        """rescales samles from U(0, 1) to the desired range"""
        delta = self.maximum - self.minimum
        return self.minimum + samples * delta
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        samples = self.rand(sample_shape, dtype=dtype, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class DeltaPrior(BasePrior):
    """Dirac Delta Prior distribution """
    
    def __init__(self, value, device = 'cpu', seed = None):
        """Dirac delta function prior, this always returns value. """
        super(DeltaPrior, self).__init__(value, value, device)
        self.value = value 
        return
        
    @property
    def name(self):
        return 'delta'
        
    @property
    def mean(self):
        return self.value
    
    @property
    def std(self):
        return 1
        
    def log_prob(self, samples):
        """Return log_prob = 0 (prob = 1)    when samples  = self.samples
        otherwise log_prob = -inf (prob = 0) when samples != self.samples
        """
        check = self.value * torch.ones(samples.shape)
        diff = abs(check - samples)
        
        log_prob = torch.zeros(samples.shape)
        #log_prob[diff !=0] = -torch.inf
        
        return log_prob

    def sample(self, sample_shape, standardize=False, dtype=None): #standardize does nothing here
        return self.value * torch.ones(sample_shape, dtype=dtype, device = self.device)
    
    
    
    
class CosinePrior(BasePrior):
    """Uniform in Cosine Prior distribution"""
    
    def __init__(self, minimum=-torch.pi / 2, maximum=torch.pi / 2, device = 'cpu', seed = None):
        """Cosine prior with bounds
        """
        super(CosinePrior, self).__init__(minimum, maximum, device, seed)
        return
    
    @property
    def name(self):
        return 'cos'

    def rescale(self, samples):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (torch.sin(self.maximum) - torch.sin(self.minimum))
        return torch.arcsin(samples / norm + torch.sin(self.minimum))
 
        
    def log_prob(self, samples):
        """Return the prior probability of samples. Defined over [-pi/2, pi/2]. """
        prob = torch.cos(samples) / 2 * self.is_in_prior_range(samples)
        return torch.log(prob)
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
    
    
class SinePrior(BasePrior):
    """Uniform in Sine Prior distribution"""
    
    def __init__(self, minimum=0, maximum=torch.pi , device = 'cpu', seed = None):
        """Sine prior with bounds
        """
        super(SinePrior, self).__init__(minimum, maximum, device, seed)
        return
    
    @property
    def name(self):
        return 'sin'

    def rescale(self, samples):
        """
        'Rescale' a sample from the unit line element to a uniform in cosine prior.

        This maps to the inverse CDF. This has been analytically solved for this case.
        """
        norm = 1 / (torch.cos(self.minimum) - torch.cos(self.maximum))
        return torch.arccos(torch.cos(self.minimum) - samples / norm)

    def log_prob(self, samples):
        """Return the log prior probability of samples. Defined over [-pi/2, pi/2]. """
        prob = torch.sin(samples) / 2 * self.is_in_prior_range(samples)
        return torch.log(prob)
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
    
class PowerLawPrior(BasePrior):
    
    def __init__(self, minimum, maximum, alpha, device = 'cpu', seed = None):
        super(PowerLawPrior, self).__init__(minimum, maximum, device, seed)
        
        self.alpha = alpha
        return
    
    @property
    def name(self):
        return 'power-law'
    
    def rescale(self, samples):
        """--- adapted from Bilby ---"""
        
        if self.alpha == -1:
            return self.minimum * torch.exp(samples * torch.log(self.maximum / self.minimum))
        else:
            return  (self.minimum ** (1 + self.alpha) + samples *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))
        


    def prob(self, samples):
        """Return the prior probability of samples
        """
        if self.alpha == -1:
            return torch.nan_to_num(1 / samples / torch.log(self.maximum / self.minimum)) * self.is_in_prior_range(samples)
        else:
            return torch.nan_to_num(samples ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(samples)

    def log_prob(self, samples):
        """Return the logarithmic prior probability of samples
        """
        if self.alpha == -1:
            normalising = 1. / torch.log(self.maximum / self.minimum)
        else:
            normalising = (1 + self.alpha) / (self.maximum ** (1 + self.alpha) -
                                              self.minimum ** (1 + self.alpha))

        
        ln_in_range = torch.log(1. * self.is_in_prior_range(samples))
        ln_p = self.alpha * torch.nan_to_num(torch.log(samples)) + torch.log(normalising)

        return ln_p + ln_in_range

    
    def sample(self, sample_shape, standardize=False, dtype=None):
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class GammaPrior(BasePrior):
    """Gamma Prior distribution
    Wrapper to torch.distributions.Gamma
    
    Args:
    -----
        concentration (float): concentration parameter
        rate (float)         : rate parameter
        scale (float)        : scaling factor. (Default: 1)
    
    Note:
    -----
        The scaling factor is used to rescale the samples from the Gamma distribution and 
        it is useful when this prior is used as an SNR distribution
    """
    
    def __init__(self, concentration=1.0, rate=1.0, scale=1, device = 'cpu', seed = None):
        self.concentration = torch.as_tensor(concentration, device = device).float()
        self.rate  = torch.as_tensor(rate,  device=device).float()
        self.scale = torch.as_tensor(scale, device=device).float()
        self.Gamma = Gamma(self.concentration, self.rate)
        
        super(GammaPrior, self).__init__(None, None, device, seed)
        return
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        
        #check shape: torch.distributions does not accept 1D samples
        if isinstance(sample_shape, int):
            sample_shape = (1, sample_shape)
            samples = self.Gamma.sample(sample_shape).squeeze()
        else:
            samples = self.Gamma.sample(sample_shape)
        
        #multiply by the scaling factor
        samples *= self.scale
        
        #eventually standardize
        if standardize:
            samples = self.standardize_samples(samples)
        
        return samples

    
    
#=======================================
# GW Prior Distributions
#=======================================    
    
class M_uniform_in_components(BasePrior):
    """Class that manages total Mass M prior from uniform distributed masses"""
    
    def __init__(self, m1, m2):
        assert isinstance(m1, UniformPrior), "m1 is not an instance of UniformPrior"
        assert isinstance(m2, UniformPrior), "m2 is not an instance of UniformPrior"
        
        self.m1 = m1
        self.m2 = m2
        minimum = float(m1.minimum + m2.minimum)
        maximum = float(m1.maximum + m2.maximum)
        super(M_uniform_in_components, self).__init__(minimum, maximum, m1.device)
        return
    
    @property
    def name(self):
        return 'M'
    
    @property
    def mean(self):
        if not hasattr(self, '_mean'):
            self._mean = self.sample(N_).mean()
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = self.sample(N_).std()
        return self._std
    
    def sample(self, sample_shape, standardize = False, dtype=None):
        m1 = self.m1.sample(sample_shape, dtype=dtype)
        m2 = self.m2.sample(sample_shape, dtype=dtype)
        M = m1+m2
        if standardize:
            M = self.standardize_samples(M)
        return M
        
        
class q_uniform_in_components(BasePrior):
    """Class that manages total Mass M prior from uniform distributed masses"""
    
    def __init__(self, m1, m2):
        assert isinstance(m1, UniformPrior), "m1 is not an instance of UniformPrior"
        assert isinstance(m2, UniformPrior), "m2 is not an instance of UniformPrior"
        
        self.m1 = m1
        self.m2 = m2
        minimum = float(m2.minimum / m1.maximum)
        maximum = float(m2.maximum / m1.maximum)
        super(q_uniform_in_components, self).__init__(minimum, maximum, m1.device)
        return
    
    @property
    def name(self):
        return 'q'
    
    @property
    def mean(self):
        if not hasattr(self, '_mean'):
            self._mean = self.sample(N_).mean()
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = self.sample(N_).std()
        return self._std
    
    @staticmethod
    def _sort_masses(m1_samp, m2_samp):
        """Sort m1 and m2 masses so that m2 <= m1"""
        m = torch.stack([m1_samp, m2_samp]).T
        sorted, _ = torch.sort(m)
        sorted_m1 = sorted.T[1]
        sorted_m2 = sorted.T[0]
        return sorted_m1, sorted_m2
    
    def sample(self, sample_shape, standardize = False, dtype=None):
        m1 = self.m1.sample(sample_shape, dtype=dtype)
        m2 = self.m2.sample(sample_shape, dtype=dtype)
        
        m1, m2 = self._sort_masses(m1, m2)

        q = m2/m1
        
        if standardize:
            q = self.standardize_samples(q)
        return q
    
    
class Mchirp_uniform_in_components(BasePrior):
    """
        Class that manages Chirp Mass Mchirp prior from uniform distributed masses
        or mass ratio and total mass..
        
        Note:
        -----
            This prior differs from the usual p(Mchirp) \propto Mchirp given that 
            we sample Mchirp from uniformly distributed masses m1, m2.
    """
    
    def __init__(self, m1=None, m2=None, M=None, q=None):
        if m1 is not None and m2 is not None:
            assert isinstance(m1, UniformPrior), "m1 is not an instance of UniformPrior"
            assert isinstance(m2, UniformPrior), "m2 is not an instance of UniformPrior"
            self.m1 = m1
            self.m2 = m2
            device = m1.device
            # Calculate minimum and maximum Mchirp for m1, m2
            minimum = float((m1.minimum * m2.minimum) ** (3/5) / (m1.minimum + m2.minimum) ** (1/5))
            maximum = float((m1.maximum * m2.maximum) ** (3/5) / (m1.maximum + m2.maximum) ** (1/5))
        
        elif M is not None and q is not None:
            assert isinstance(M, UniformPrior), "M is not an instance of UniformPrior"
            assert isinstance(q, UniformPrior), "q is not an instance of UniformPrior"
            self.M = M
            self.q = q
            device = M.device
            #minimum/maximum will be estimated numerically when sampling
            minimum, maximum = None, None
        
        else:
            raise ValueError("Please provide either (m1, m2) or (M, q)")
        
        super(Mchirp_uniform_in_components, self).__init__(minimum, maximum, device)
        return
    
    @property
    def name(self):
        return 'Mchirp'
    
    @property
    def mean(self):
        if not hasattr(self, '_mean'):
            self._mean = self.sample(N_).mean()
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = self.sample(N_).std()
        return self._std
    
    def sample(self, sample_shape, standardize = False, dtype=None):
        if hasattr(self, 'm1'):
            m1 = self.m1.sample(sample_shape, dtype=dtype)
            m2 = self.m2.sample(sample_shape, dtype=dtype)
            Mchirp = (m1*m2)**(3/5)/(m1+m2)**(1/5)
        else:
            M = self.M.sample(sample_shape, dtype=dtype)
            q = self.q.sample(sample_shape, dtype=dtype)
            Mchirp = M * q**(3/5) / (1+q)**(6/5)

        if standardize:
            Mchirp = self.standardize_samples(Mchirp)
        return Mchirp
    
    
prior_dict_ = {'uniform'  : UniformPrior, 
               'delta'    : DeltaPrior, 
               'cos'      : CosinePrior,
               'sin'      : SinePrior, 
               'power-law': PowerLawPrior, 
               'gamma'    : GammaPrior,
               'M'        : M_uniform_in_components, 
               'q'        : q_uniform_in_components,
               'Mchirp'   : Mchirp_uniform_in_components}


#=======================================
# Multivariate Prior
#=======================================  
class MultivariatePrior():
    """
    Class that manages a multivariate (i.e. multiparameter) Prior
    with each of the parameters having its own prior
    
    Args:
    -----
        prior_dict : dict
            dictionary containing Prior distribution instances
    
    """
    
    def __init__(self, prior_dict, seed=None, device='cpu'):
        
        """
        Constructor Args:
        -----------------
            prior_dict (dict): dictionary containing Prior distribution instances / dictionary metadata
            seed (int): seed for reproducibility
            device (str): either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')        
        """
        
        self.prior_dict = prior_dict
        self.seed   = seed
        self.device = device
        
        self.priors = self._load_priors(self.prior_dict, self.seed, self.device)
        self.names  = list(self.prior_dict.keys())

        return
    
    @staticmethod
    def _load_priors(prior_dict, seed, device):
        """Loads the priors from a dictionary of prior distributions"""
        priors = dict()
        
        for i, par in enumerate(prior_dict):
            if isinstance(prior_dict[par], BasePrior):
                priors[par] = prior_dict[par]
                continue

            dist_name = prior_dict[par]['distribution']
            kwargs = prior_dict[par]['kwargs']
            #evaluate string expressions in kwargs
            kwargs = {key: eval(value) if isinstance(value, str) else value for key, value in kwargs.items()}
            kwargs['device'] = device

            if seed is not None:
                if isinstance(seed, dict):
                    kwargs['seed'] = seed[par] + i
                else:
                    kwargs['seed'] = seed + i

            priors[par] = prior_dict_[dist_name](**kwargs)

        return priors
    
    def add_prior(self, new_prior_dict, seed=None):
        
        prior_dict = self.prior_dict.copy()
        prior_dict.update(new_prior_dict)    
        self.__init__(prior_dict, seed, self.device)
        
        return self
    
    def log_prob(self, samples):
        """Samples must be a dictionary containing a set of parameter samples of shape [Nbatch, 1]"""
        logP = torch.stack([self.priors[name].log_prob(samples[name]) for name in samples.keys()], dim = -1)
        return logP.sum(-1)
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        samples = dict()
        for name in self.names:
            samples[name] = self.priors[name].sample(sample_shape, standardize, dtype=dtype)
        samples = TensorDict.from_dict(samples)
        return samples
            
        
        


    