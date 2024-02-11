import torch

__all__ = ['UniformPrior', 'DeltaPrior', 'CosinePrior', 'SinePrior', 'PowerLawPrior', 'MultivariatePrior']


class BasePrior(object):
    """Base class for Prior Distributions"""
    
    def __init__(self, minimum=None, maximum=None , device = 'cpu'):

        assert maximum >= minimum, f" high must be >= than low, given {maximum} and {minimum} respectively"

        self.device  = device
        self.minimum = torch.tensor(minimum).to(device)
        self.maximum = torch.tensor(maximum).to(device)
        return
    
    @property
    def mean(self):
        #print('default')
        if not hasattr(self, '_mean'):
            self._mean = self.sample((10_000)).mean()
        return self._mean
    
    @property
    def std(self):
        if not hasattr(self, '_std'):
            self._std = self.sample((10_000)).std()
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

    def __init__(self, minimum, maximum, device = 'cpu'):
        super(UniformPrior, self).__init__(minimum, maximum, device)
        return
    
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
    
    def sample(self, sample_shape, standardize=False):
        samples = torch.rand(sample_shape, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class DeltaPrior(BasePrior):
    """Dirac Delta Prior distribution """
    
    def __init__(self, value, device = 'cpu'):
        """Dirac delta function prior, this always returns value. """
        super(DeltaPrior, self).__init__(value, value, device)
        self.value = value
        return
        
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
        log_prob[diff !=0] = -torch.inf
        
        return log_prob

    def sample(self, sample_shape, standardize=False): #standardize does nothing here
        return self.value * torch.ones(sample_shape)
    
    
    
    
class CosinePrior(BasePrior):
    """Uniform in Cosine Prior distribution"""
    
    def __init__(self, minimum=-torch.pi / 2, maximum=torch.pi / 2, device = 'cpu'):
        """Cosine prior with bounds
        """
        super(CosinePrior, self).__init__(minimum, maximum, device)
        return
    

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
    
    def sample(self, sample_shape, standardize=False):
        samples = torch.rand(sample_shape, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
    
    
class SinePrior(BasePrior):
    """Uniform in Sine Prior distribution"""
    
    def __init__(self, minimum=0, maximum=torch.pi , device = 'cpu'):
        """Cosine prior with bounds
        """
        super(SinePrior, self).__init__(minimum, maximum, device)
        return

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
    
    def sample(self, sample_shape, standardize=False):
        samples = torch.rand(sample_shape, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
    
class PowerLawPrior(BasePrior):
    
    def __init__(self, minimum, maximum, alpha, device = 'cpu'):
        super(PowerLawPrior, self).__init__(minimum, maximum, device)
        
        self.alpha = alpha
        return
    
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

    
    def sample(self, sample_shape, standardize=False):
        samples = torch.rand(sample_shape, device = self.device)
        samples = self.rescale(samples)
        if standardize:
            samples = self.standardize_samples(samples)
        return samples
    
    
class MultivariatePrior():
    """
    Class that manages a multivariate (i.e. multiparameter) Prior
    with each of the parameters having its own prior
    
    Args:
    -----
        prior_dict : dict
            dictionary containing Prior distribution instances
    
    """
    
    def __init__(self, prior_dict):
        
        self.priors = prior_dict
        self.names  = list(prior_dict.keys())
        
        return
    
    def log_prob(self, samples):
        """Samples must be a dictionary containing a set of parameter samples of shape [Nbatch, 1]"""
        '''
        for name in samples.keys():
            logP = self.priors[name].log_prob(samples[name])
            print(name, logP)
            print(name, self.priors[name], self.priors[name].minimum, self.priors[name].maximum)
        '''
        logP = torch.cat([self.priors[name].log_prob(samples[name]) for name in samples.keys()], dim = 1)
        return logP.sum(-1)
    
    def sample(self, sample_shape, standardize=False):
        samples = dict()
        for name in self.names:
            samples[name] = self.priors[name].sample(sample_shape, standardize)
        return samples
            
        

        
    
    

   


    
    







if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    
    
    u = UniformPrior(minimum=10, maximum=100)
    u = CosinePrior()
    u = PowerLawPrior(minimum=100, maximum=6000, alpha=2)
    #u = DeltaPrior(123)
    #print(u.mean)
        
    s = u.sample((100,1), standardize=False)
    #plt.hist(s.numpy(), 100);
    #plt.show()
    
    
    x = UniformPrior(1, 8000).sample(s.shape)
    log_prob = u.log_prob(x)
    
    
    print(log_prob, log_prob.shape)
    