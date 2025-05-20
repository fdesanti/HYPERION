import torch

from ..types import TensorSamples
from torch.distributions import Gamma

N_ = int(1e7)

#=======================================
# Rand Wrapper Class
#=======================================
class Rand():
    """
    Wrapper class to torch.rand to explot a random generator with a fixed seed
    for reproducibility. 

    Args:
        seed   (int): Seed for reproducibility. (Default: None)
        device (str): Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
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
    """
    Base class for Prior Distributions
    
    Args:
        minimum (float): Minimum value of the prior distribution
        maximum (float): Maximum value of the prior distribution
        device  (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed    (int)  : Seed for reproducibility. (Default: None)
    """
    def __init__(self, minimum=None, maximum=None, device='cpu', seed=None):

        self.device  = device
        self.rand    = Rand(seed, device)
        self.minimum = minimum
        self.maximum = maximum
        assert self.maximum >= self.minimum, f"maximum must be >= than minimum, given {maximum} and {minimum} respectively"
    
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
        """
        Returns 1 if samples is in the prior boundaries, 0 otherwise
        """
        return ( (samples >= self.minimum) & (samples <= self.maximum) ).bool().int()
    
    def standardize_samples(self, samples):
        """
        Standardize samples to have zero mean and unit variance
        """
        return (samples - self.mean) / self.std
    
    def de_standardize_samples(self, samples):
        """
        De-standardize samples to the original range
        """
        return (samples * self.std) + self.mean
    
    def log_prob(self, samples):
        raise NotImplementedError
    
    def rescale(self, sample_shape):
        """
        Rescale U(0, 1) samples to the desired range and prior
        """
        raise NotImplementedError
    
    def sample(self, sample_shape):
        raise NotImplementedError
    

class UniformPrior(BasePrior):
    """
    Uniform prior distribution class

    Args:
        minimum (float): Minimum value of the prior distribution
        maximum (float): Maximum value of the prior distribution
        device  (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed    (int)  : Seed for reproducibility. (Default: None)
    """

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
        """
        Returns the log prior probability of samples
        """
        return torch.xlogy(1, (samples >= self.minimum) & (samples <= self.maximum)) - torch.xlogy(1, self.maximum - self.minimum)

    def rescale(self, samples):
        """rescales samles from U(0, 1) to the desired range"""
        delta = self.maximum - self.minimum
        return self.minimum + samples * delta
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        """
        Sample from the uniform prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)

        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        samples = self.rand(sample_shape, dtype=dtype, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class DeltaPrior(BasePrior):
    """
    Dirac Delta Prior distribution

    .. math::
        p(x) = \delta(x - value)
    
    Args:
        value  (float): Value of the Dirac delta function
        device (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed   (int)  : Seed for reproducibility. (Default
    
    """
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
        """
        Sample from the delta prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Not used. (Default: False)
        
        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        return self.value * torch.ones(sample_shape, dtype=dtype, device = self.device)
    

class CosinePrior(BasePrior):
    """
    Uniform in Cosine Prior distribution. 

    Args:
        minimum (float): Minimum value of the prior distribution. (Default: -π/2)
        maximum (float): Maximum value of the prior distribution. (Default: π/2)
        device  (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed    (int)  : Seed for reproducibility. (Default: None)

    Note:
        This prior might be used for the declination angle of a source in the sky.
    """
    
    def __init__(self, minimum=-torch.pi / 2, maximum=torch.pi / 2, device = 'cpu', seed = None):
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
        """
        Sample from the prior distribution.

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)

        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
class SinePrior(BasePrior):
    """
    Uniform in Sine Prior distribution
    
    Args:
        minimum (float): Minimum value of the prior distribution. (Default: 0)
        maximum (float): Maximum value of the prior distribution. (Default: π)
        device  (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed    (int)  : Seed for reproducibility. (Default: None)

    Note:
        This prior might be used for the inclination angle of a source in the sky.
    """
    
    def __init__(self, minimum=0, maximum=torch.pi , device = 'cpu', seed = None):
        super(SinePrior, self).__init__(minimum, maximum, device, seed)
    
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
        """Return the log prior probability of samples. Defined over [-π/2, π/2]. """
        prob = torch.sin(samples) / 2 * self.is_in_prior_range(samples)
        return torch.log(prob)
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        """
        Sample from the prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)
        
        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
class PowerLawPrior(BasePrior):
    r"""
    Power Law Prior distribution

    .. math::

        p(x) \propto x^{\alpha}

    Args:
        minimum (float): Minimum value of the prior distribution
        maximum (float): Maximum value of the prior distribution
        alpha   (float): Power law index
        device  (str)  : Either "cpu" or "cuda" for the device to run the code. (Default: 'cpu')
        seed    (int)  : Seed for reproducibility. (Default: None)
    """
    
    def __init__(self, minimum, maximum, alpha, device = 'cpu', seed = None):
        super(PowerLawPrior, self).__init__(minimum, maximum, device, seed)
        
        self.alpha = alpha
        
    @property
    def name(self):
        return 'power-law'
    
    def rescale(self, samples):
        if self.alpha == -1:
            return self.minimum * torch.exp(samples * torch.log(self.maximum / self.minimum))
        else:
            return  (self.minimum ** (1 + self.alpha) + samples *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))
        
    def prob(self, samples):
        """
        Return the prior probability of samples
        """
        if self.alpha == -1:
            return torch.nan_to_num(1 / samples / torch.log(self.maximum / self.minimum)) * self.is_in_prior_range(samples)
        else:
            return torch.nan_to_num(samples ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(samples)

    def log_prob(self, samples):
        """
        Return the logarithmic prior probability of samples
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
        """
        Sample from the power law prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)
        
        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        samples = self.rand(sample_shape, device = self.device, dtype=dtype)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class GammaPrior(BasePrior):
    """
    Gamma Prior distribution (wrapper to torch.distributions.Gamma)
    
    Args:
        concentration (float): concentration parameter
        rate (float)         : rate parameter
        scale (float)        : scaling factor. (Default: 1)
    
    Note:
        The scaling factor is used to rescale the samples from the Gamma distribution and it is useful when this prior is used as an SNR distribution
    """
    
    def __init__(self, concentration=1.0, rate=1.0, scale=1, device = 'cpu', seed = None):
        self.concentration = torch.as_tensor(concentration, device = device).float()
        self.rate  = torch.as_tensor(rate,  device=device).float()
        self.scale = torch.as_tensor(scale, device=device).float()
        self.Gamma = Gamma(self.concentration, self.rate)
        
        super(GammaPrior, self).__init__(None, None, device, seed)
    
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
    r"""
    Class that manages total Mass M prior from uniform distributed masses

    .. math::
        M = m_1 + m_2
    
    where m1 and m2 are the component masses of the binary system.
    
    Args:
        m1 (UniformPrior): Prior distribution of mass m1
        m2 (UniformPrior): Prior distribution of mass m2
    """
    
    def __init__(self, m1, m2, **kwargs):
        assert isinstance(m1, UniformPrior), "m1 is not an instance of UniformPrior"
        assert isinstance(m2, UniformPrior), "m2 is not an instance of UniformPrior"
        
        self.m1 = m1
        self.m2 = m2
        minimum = float(m1.minimum + m2.minimum)
        maximum = float(m1.maximum + m2.maximum)
        super(M_uniform_in_components, self).__init__(minimum, maximum, m1.device)
    
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
        """
        Sample from the total mass prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)

        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        m1 = self.m1.sample(sample_shape, dtype=dtype)
        m2 = self.m2.sample(sample_shape, dtype=dtype)
        M = m1+m2
        if standardize:
            M = self.standardize_samples(M)
        return M
        
        
class q_uniform_in_components(BasePrior):
    r"""
    Class that manages total Mass M prior from uniform distributed masses
    
    .. math::

        q = m_2 / m_1 \quad \text{with} \quad m_2 \leq m_1

    where m1 and m2 are the component masses of the binary system.

    Args:
        m1 (UniformPrior): Prior distribution of mass m1
        m2 (UniformPrior): Prior distribution of mass m2
    """
    def __init__(self, m1, m2, minimum=None, maximum=None, **kwargs):
        assert isinstance(m1, UniformPrior), "m1 is not an instance of UniformPrior"
        assert isinstance(m2, UniformPrior), "m2 is not an instance of UniformPrior"
        
        self.m1 = m1
        self.m2 = m2
        minimum = float(m2.minimum / m1.maximum)
        maximum = float(m2.maximum / m1.maximum)
        super(q_uniform_in_components, self).__init__(minimum, maximum, m1.device)
    
    
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
        """
        Sample from the mass ratio prior

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)
        
        Returns:
            torch.Tensor: Samples from the prior distribution
        """
        m1 = self.m1.sample(sample_shape, dtype=dtype)
        m2 = self.m2.sample(sample_shape, dtype=dtype)
        
        m1, m2 = self._sort_masses(m1, m2)

        q = m2/m1
        
        if standardize:
            q = self.standardize_samples(q)
        return q
    
    
class Mchirp_uniform_in_components(BasePrior):
    r"""
    Class that manages Chirp Mass Mchirp prior from uniform distributed masses
    or mass ratio and total mass.

    .. math::

        \mathcal{M} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}

    where m1 and m2 are the component masses of the binary system.

    Args:
        m1 (UniformPrior): Prior distribution of mass m1
        m2 (UniformPrior): Prior distribution of mass m2
    
    Note:
        This prior differs from the usual 
        
        .. math::
        
            p(\mathcal{M}) \propto \mathcal{M} 
        
        given that we sample Mchirp from uniformly distributed masses m1, m2.
    """
    
    def __init__(self, m1=None, m2=None, M=None, q=None, **kwargs):
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
        """
        Sample from the chirp mass prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)

        Returns:
            torch.Tensor: Samples from the prior distribution
        """
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
        prior_dict (dict): Dictionary containing BasePrior distribution instances
        seed        (int): Seed for reproducibility
        device      (str): Either "cpu" or "cuda" for the device to run the code. (Default
    
    """
    
    def __init__(self, prior_dict, seed=None, device='cpu'):
        self.prior_dict = prior_dict
        self.seed   = seed
        self.device = device
        
        self.priors = self._load_priors(self.prior_dict, self.seed, self.device)
        self.names  = list(self.prior_dict.keys())
    
    @staticmethod
    def _load_priors(prior_dict, seed, device):
        """
        Loads the priors from a dictionary of prior distributions
        """
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
    
    @property
    def metadata(self):
        """
        Returns:
            Dictionary containing the prior metadata:
                - **priors**: dictionary containing the (BasePrior) distributions
                - **means**: dictionary containing the means of the priors
                - **stds**: dictionary containing the standard deviations of the priors
                - **bounds**: dictionary containing the bounds of the priors [minimum, maximum]
                - **inference_parameters**: list of the names of the prior parameters
        """
        if not hasattr(self, '_metadata'):
            self._metadata = self._get_prior_metadata()
        return self._metadata

    def _get_prior_metadata(self):
        """Returns a dictionary containing the prior metadata"""
        prior_metadata = dict()
        prior_metadata['priors'] = self.priors
        prior_metadata['means']  = {name: prior.mean for name, prior in self.priors.items()}
        prior_metadata['stds']   = {name: prior.std  for name, prior in self.priors.items()}
        prior_metadata['bounds'] = {name: (prior.minimum, prior.maximum) for name, prior in self.priors.items()}
        prior_metadata['inference_parameters'] = self.names
        return prior_metadata
    
    def add_prior(self, new_prior_dict, seed=None):
        """
        Adds a new prior to the current prior dictionary

        Args:
            new_prior_dict (dict): Dictionary containing the new prior distribution
            seed            (int): Seed for reproducibility. (Default: None)

        Returns:
            MultivariatePrior: New instance of the MultivariatePrior class
        """
        prior_dict = self.prior_dict.copy()
        prior_dict.update(new_prior_dict)    
        self.__init__(prior_dict, seed, self.device)
        return self
    
    def standardize_samples(self, samples):
        """
        Standardizes samples to zero mean and unit variance
        
        Args:
            samples (TensorSamples): Dictionary containing the parameter samples

        Returns:
            TensorSamples: Standardized samples
        """
        for key in samples.keys():
            samples[key] = self.priors[key].standardize_samples(samples[key])
        return samples
    
    def de_standardize_samples(self, samples):
        """
        De-standardizes samples to the original range

        Args:
            samples (TensorSamples): Dictionary containing the parameter samples
        
        Returns:
            TensorSamples: De-standardized samples
        """
        for key in samples.keys():
            samples[key] = self.priors[key].de_standardize_samples(samples[key])
        return samples

    def log_prob(self, samples):
        """
        Returns the log prior probability of samples

        Args:
            samples (TensorSamples): Dictionary containing the parameter samples
        
        Returns:
            torch.Tensor: Log prior probability of the samples
        """
        logP = torch.stack([self.priors[name].log_prob(samples[name]) for name in samples.keys()], dim = -1)
        return logP.sum(-1)
    
    def sample(self, sample_shape, standardize=False, dtype=None):
        """
        Sample from the multivariate prior distribution

        Args:
            sample_shape (int, tuple): Shape of the samples
            standardize        (bool): Standardize the samples to have zero mean and unit variance. (Default: False)
            dtype       (torch.dtype): Data type of the samples. (Default: None)

        Returns:
            TensorSamples: Samples from the prior distribution
        """
        samples = dict()
        for name in self.names:
            samples[name] = self.priors[name].sample(sample_shape, standardize, dtype=dtype)
        samples = TensorSamples.from_dict(samples)
        return samples