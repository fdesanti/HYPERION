import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal as torchNormal
from torch.distributions import VonMises as torchVonMises
from torch.distributions import MultivariateNormal as torchMultivariateNormal


class MultivariateNormalBase(nn.Module):
    """Multivariate Normal the base distribution for the flow. 
       Exploits the torch.distributions module
       The distribution is initialized to zero mean and unit variance
    
        Args:
            dim : int  
                Number of dimensions (i.e. of physical inference parameters). (Default: 10)

            trainable : bool
                Whether to train the parameters (means and stds) of the distribution during training. (Default: False)
                
            
        Methods:
            - log_prob: returns the log_prob given samples of dim (N batches, self.dim)
            
            - sample: samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
    def __init__(self, 
                 dim        :int  =  10,
                 trainable  :bool = False,
                 ):
        super(MultivariateNormalBase, self).__init__()
        self.dim = dim
        self.trainable = trainable
        
        self.initialize_distribution()
        return
    
    def initialize_distribution(self):
        """Initializes the distributions given the paramters from the __init__"""
        
        if self.trainable:
            self.mean = nn.Parameter(torch.randn((self.dim)))
            self.var  = nn.Parameter(torch.eye(self.dim))
            #self.mean = nn.Parameter(torch.zeros(self.dim))
            #self.var  = nn.Parameter(torch.randint(1, 6, (self.dim,)) * torch.eye(self.dim))
        else:
            self.register_buffer("mean", torch.zeros(self.dim))
            self.register_buffer("var",  torch.eye(self.dim))

    @property
    def MultivariateNormal(self):
        return torchMultivariateNormal(self.mean, self.var, validate_args = False)
    
    def log_prob(self, samples, embedded_strain=None):
        """assumes that samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if samples.shape[1] != self.dim:
            raise ValueError(f'Wrong samples dim. Expected (batch_size, {self.dim}) and got {samples.dim}')
        
        return self.MultivariateNormal.log_prob(samples)
    
    def sample(self, num_samples, embedded_strain=None):
        
        #by default .samples returns [1, num_samples, dim] so we delete 1 dimension
        return self.MultivariateNormal.sample((1,num_samples)).squeeze(0)
    

class ConditionalMultivariateNormalBase(nn.Module):
    """Multivariate Normal the base distribution for the flow. 
       Exploits the torch.distributions module
       The distribution is initialized to zero mean and unit variance
    
        Args:
            dim : int  
                Number of dimensions (i.e. of physical inference parameters). (Default: 10)

            trainable : bool
                Whether to train the parameters (means and stds) of the distribution during training. (Default: False)
                
            
        Methods:
            - log_prob: returns the log_prob given samples of dim (N batches, self.dim)
            
            - sample: samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
    def __init__(self, 
                 dim        :int  =  10,
                 neural_network_kwargs={},
                 ):
        super(ConditionalMultivariateNormalBase, self).__init__()
        self.dim = dim      
        
        activation = neural_network_kwargs.get('activation', nn.ELU())
        dropout    = neural_network_kwargs.get('dropout', 0.2)
        layer_dim  = neural_network_kwargs.get('layer_dims', 256)
        
        self.mean_network = nn.Sequential(nn.LazyLinear(layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, self.dim), 
                                            activation)
        
        self.var_network = nn.Sequential(nn.LazyLinear(layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, self.dim), 
                                            nn.Softplus())
        
        self.eps = 1e-6
    
        return
    
    
    
    def log_prob(self, samples, embedded_strain):
        """assumes that samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if samples.shape[1] != self.dim:
            raise ValueError(f'Wrong samples dim. Expected (batch_size, {self.dim}) and got {samples.dim}')
        
        mean = self.mean_network(embedded_strain)
        var  = self.var_network(embedded_strain) + self.eps
        
        return torchMultivariateNormal(mean, torch.diag_embed(var)).log_prob(samples)
    
    def sample(self, num_samples, embedded_strain):
        
        #compute the mean and variance
        #NB here we assume embedded_strain has dim [1, strain_dim] (i.e. 1 sample per batch)
        mean = self.mean_network(embedded_strain).squeeze(0)
        var  = self.var_network(embedded_strain).squeeze(0) + self.eps

        #by default .samples returns [1, num_samples, dim] so we delete 1 dimension
        return torchMultivariateNormal(mean, torch.diag_embed(var)).sample((1,num_samples)).squeeze(0)
    

class MultivariateGaussianMixtureBase(nn.Module):
    """Mixture of Multivariate Normal the base distribution for the flow."""

    def __init__(self, 
                 num_components :int  =  2,
                 dim            :int  =  10,
                 trainable      :bool = True,
                 ):
        super(MultivariateGaussianMixtureBase, self).__init__()
        self.num_components = num_components
        self.dim            = dim
        self.trainable      = trainable

        #initialize the mixture components
        self.mixture_components = nn.ModuleList([MultivariateNormalBase(dim, trainable) for _ in range(num_components)])
        
        #initialize the mixture weights
        mixture_weights = torch.ones(num_components)/num_components
        if trainable:
            self.mixture_weights = nn.Parameter(torch.log(mixture_weights))
        else:
            self.register_buffer("mixture_weights", mixture_weights)
        return
    
    def log_prob(self, samples, embedded_strain=None):
        """assumes that samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if samples.shape[1] != self.dim:
            raise ValueError(f'Wrong samples dim. Expected (batch_size, {self.dim}) and got {samples.dim}')
        
        weights  = F.softmax(self.mixture_weights)

        log_prob = [weights[i].log() + self.mixture_components[i].log_prob(samples) for i in range(self.num_components)]
        
        return torch.sum(torch.stack(log_prob, axis=0), axis=0)
    
    def sample(self, num_samples, embedded_strain=None):
        #sample the component
        weights  = F.softmax(self.mixture_weights, dim=0)
        component_samples = torch.multinomial(weights, num_samples, replacement = True)
        
        #sample from the components
        samples = torch.empty((num_samples, self.dim), device = self.mixture_components[0].mean.device)
        
        for i in range(self.num_components):
            mask = component_samples == i
            samples[mask, :] = self.mixture_components[i].sample(mask.sum())

        return samples

class ConditionalMultivariateGaussianMixtureBase(nn.Module):
    """Conditional Mixture of Multivariate Normal the base distribution for the flow."""
    def __init__(self, 
                 dim            :int   =  10,
                 num_components :int   =   2,
                 neural_network_kwargs =  {},
                 ):
        super(ConditionalMultivariateGaussianMixtureBase, self).__init__()
        self.dim            = dim
        self.num_components = num_components
        
        activation = neural_network_kwargs.get('activation', nn.ELU())
        dropout    = neural_network_kwargs.get('dropout', 0.2)
        layer_dim  = neural_network_kwargs.get('layer_dims', 256)
        
        #initialize the mixture components
        self.mixture_components = nn.ModuleList([ConditionalMultivariateNormalBase(dim, 
                                                                                   trainable=True, 
                                                                                   neural_network_kwargs=neural_network_kwargs) 
                                                 for _ in range(num_components)])
        
        self.weights_network = nn.Sequential(nn.LazyLinear(layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, layer_dim),
                                            activation,
                                            nn.Dropout(dropout),
                                            nn.Linear(layer_dim, num_components), 
                                            nn.Softmax(dim=1))
    
        return
    
    
    
    def log_prob(self, samples, embedded_strain):
        """assumes that samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if samples.shape[1] != self.dim:
            raise ValueError(f'Wrong samples dim. Expected (batch_size, {self.dim}) and got {samples.dim}')
        
        weights  = self.weights_network(embedded_strain)

        log_prob = [weights[i].log() + self.mixture_components[i].log_prob(samples) for i in range(self.num_components)]
        
        return torch.sum(torch.stack(log_prob, axis=0), axis=0)
    
    def sample(self, num_samples, embedded_strain=None):
        #sample the component
        weights  = self.weights_network(embedded_strain)
        component_samples = torch.multinomial(weights, num_samples, replacement = True)
        
        #sample from the components
        samples = torch.empty((num_samples, self.dim), device = self.mixture_components[0].mean.device)
        
        for i in range(self.num_components):
            mask = component_samples == i
            samples[mask, :] = self.mixture_components[i].sample(mask.sum())

        return samples


class VonMisesNormal(nn.Module):
    """It implements the base distribution for the flow. 
       Exploits the torch.distributions module
    
        Args:
            - parameters (dict of tuples): dict containing tuples of paramters for the distributions. These needs to be float
                                          (default {'Normal': (0.0, 1.0), 'VonMises': (0.0, 1.0)})
            - dim (dict of integers):  dict specifying the dim of each distributions.
                                        (default: {'Normal': 6, 'VonMises': 4} for the CE case)
            
            - device (torch.device): used to specify the device (cuda or cpu)  
            
        Methods:
            - log_prob: returns the log_prob given samples of dim (N batches, Normal_dim + VonMises_dim, Num_samples)
            
            - sample: samples from the prior distribution returning a tensor of dim (Normal_dim + VonMises_dim, Num_samples)
    
    """
    
    def __init__(self, 
                 parameters = {'Normal': [0.0, 1.0], 'VonMises': [0.0, 1.0]},
                 dim        = {'Normal': 8, 'VonMises': 0},
                 trainable  = False,
                 ):
        super(VonMisesNormal, self).__init__()

        #ASSERTION CHECKS ---------------------------------------------
        assert (parameters is not None) and dim is not None, 'Must specify <parameters> and <dim>'
        assert len(parameters) == len(dim), 'Number of parameters and dim does not match: found %d distributions and %d dim'%(len(parameters), len(dim))
        assert len(dim) == 2, 'I can only accept 2 distribution instances'
        
        if not all(isinstance(parameters[item], list) for item in parameters):
            raise ValueError('Expected <parameters> to be a list of tuples, got', parameters)
        
        if not all(isinstance(dim[item], int) for item in dim):
            raise ValueError('Expected <dim> to be a list of integers')
        #----------------------------------------------------------------

        
        self.parameters   = parameters
        self.Normal_dim   = dim['Normal']
        self.VonMises_dim = dim['VonMises']
        
        #DISTRIBUTION INITIALIZATION --------------------------------
        self.initialize_distributions(trainable)
        
        self.Normal_mask, self.VonMises_mask = self._create_Normal_and_VonMises_masks()
        return
    
    
    def initialize_distributions(self, trainable):
        """Initializes the distributions given the paramters from the __init__"""
        #Normal
        if trainable:
            self.norm_mean = nn.Parameter(torch.zeros(self.Normal_dim))
            self.norm_var  = nn.Parameter(torch.ones(self.Normal_dim))
        else:
            self.register_buffer("norm_mean", torch.tensor(self.parameters['Normal'][0]))
            self.register_buffer("norm_var",  torch.tensor(self.parameters['Normal'][1]))

        #VonMises
        if trainable:
            self.von_mises_mean = nn.Parameter(torch.zeros(self.VonMises_dim))
            self.concentration  = nn.Parameter(torch.ones(self.VonMises_dim))
        else:
            self.register_buffer("von_mises_mean", torch.tensor(self.parameters['VonMises'][0]))
            self.register_buffer("concentration",  torch.tensor(self.parameters['VonMises'][1]))


    @property
    def Normal(self):
        return torchNormal(self.norm_mean, self.norm_var)

    @property
    def VonMises(self):
        return torchVonMises(self.von_mises_mean, self.concentration)
    
    
    def _create_Normal_and_VonMises_masks(self):
        true          = torch.ones(self.Normal_dim, dtype = torch.bool)
        false         = torch.zeros(self.VonMises_dim, dtype = torch.bool)
        Normal_mask   = torch.cat([true, false], axis = 0)
        VonMises_mask = ~Normal_mask
        return Normal_mask, VonMises_mask
        
        
    def log_prob(self, samples, embedded_strain=None):
        """assumes that samples have dim (Nbatch, N_norm+N_vonmises) ie 1 sample per batch"""
        
        if samples.dim[1] != self.Normal_dim+self.VonMises_dim:
            raise ValueError("Wrong samples dim. Expected (None, %d, None) and got %s"%((self.Normal_dim+self.VonMises_dim), samples.dim))
        
        #print('>>>>>>>>>> samples dim', samples.dim)
        xn = samples[:, self.Normal_mask]
        xv = samples[:, self.VonMises_mask]
        
        
        log_prob = [ self.Normal.log_prob(xn), self.VonMises.log_prob(xv) ]   
        
        log_prob = torch.cat(log_prob, axis = 1 )
        
                
        return torch.sum(log_prob, dim=(1))
    
    
    def sample(self, num_samples, embedded_strain=None):
        
        #Norm_samples      = torch.fmod(self.Normal.sample((self.Normal_dim , num_samples)), 2).T
        #VonMises_samples  = torch.fmod(self.VonMises.sample((self.VonMises_dim,  num_samples)), 3).T

        Norm_samples      = self.Normal.sample((self.Normal_dim, num_samples)).T
        VonMises_samples  = self.VonMises.sample((self.VonMises_dim, num_samples)).T

        
        samples = torch.empty((num_samples, self.Normal_dim+self.VonMises_dim), device = self.norm_mean.device) #has dim [num samples, n CE parameters ]
        
        samples[:, self.Normal_mask]   = Norm_samples
        samples[:, self.VonMises_mask] = VonMises_samples
        
        return samples

base_distributions_dict = {'MultivariateNormalBase'                    : MultivariateNormalBase,
                           'ConditionalMultivariateNormalBase'         : ConditionalMultivariateNormalBase,
                           'MultivariateGaussianMixtureBase'           : MultivariateGaussianMixtureBase,
                           'ConditionalMultivariateGaussianMixtureBase': ConditionalMultivariateGaussianMixtureBase,
                           'VonMisesNormal'                            : VonMisesNormal}