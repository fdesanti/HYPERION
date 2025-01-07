import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal as torchNormal
from torch.distributions import VonMises as torchVonMises
from torch.distributions import MultivariateNormal as torchMultivariateNormal

# -------------------------
# Multivariate Gaussian
# -------------------------
class MultivariateNormalBase(nn.Module):
    """Multivariate Normal the base distribution for the flow. 
       Exploits the torch.distributions module
       The distribution is initialized to zero mean and unit variance
    
        Args:
        ------
            dim (int)        : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            trainable (bool) : Whether to train the parameters (means and stds) of the distribution during training. (Default: False)
                
            
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
    def __init__(self, 
                 dim        =  10,
                 mean       =  None,
                 var        =  None,
                 trainable  = False,
                 
                 ):
        super(MultivariateNormalBase, self).__init__()
        self.dim = dim
        self.trainable = trainable

        if mean is None: mean = torch.zeros(dim)
        if var  is None: var  = torch.eye(dim)
        
        self.initialize_distribution(mean, var)
        return
    
    def initialize_distribution(self, mean, var):
        """Initializes the distributions given the paramters from the __init__"""
        
        if self.trainable:
            self.mean = nn.Parameter(mean)
            self.var  = nn.Parameter(var)
        else:
            self.register_buffer("mean", mean)
            self.register_buffer("var",  var)

    @property
    def MultivariateNormal(self):
        return torchMultivariateNormal(self.mean, scale_tril=self.var, validate_args = False)
    
    def log_prob(self, z_samples, embedded_strain=None):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong z_samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        return self.MultivariateNormal.log_prob(z_samples)
    
    def sample(self, num_samples, embedded_strain=None):
        
        #by default .samples returns [1, num_samples, dim] so we delete 1 dimension
        return self.MultivariateNormal.sample((1,num_samples)).squeeze(0)
    
    
# ----------------------------------
# Multivariate Gaussian Conditional
# ----------------------------------
class ConditionalMultivariateNormalBase(nn.Module):
    """Multivariate Normal the base distribution for the flow conditioned on embedded context. 
       Exploits the torch.distributions module
       The distribution is initialized to zero mean and unit variance. 
       Both mean and variance are output of FC Networks conditioned on the context.
    
        Args:
        -----
            dim (int)                    : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            neural_network_kwargs (dict) : arguments to be passed to the neural network(s) that conditions the distribution.
                
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
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
        
        self.mean_network = nn.Sequential(nn.LazyLinear(layer_dim), activation,
                                          nn.Dropout(dropout),
                                          nn.Linear(layer_dim, layer_dim), activation,
                                          nn.Dropout(dropout),
                                          nn.Linear(layer_dim, self.dim), activation)
        
        self.var_network  = nn.Sequential(nn.LazyLinear(layer_dim), activation,
                                          nn.Dropout(dropout),
                                          nn.Linear(layer_dim, layer_dim), activation,
                                          nn.Dropout(dropout),
                                          nn.Linear(layer_dim, self.dim), nn.Softplus())
        
        self.eps = 1e-6
    
        return
    
    @property
    def mean(self):
        return self._mean
    @mean.setter
    def mean(self, value):
        self._mean = value
        
    @property
    def var(self):
        return self._var
    @var.setter
    def var(self, value):
        self._var = value


    def log_prob(self, z_samples, embedded_strain):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong z_samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        mean = self.mean_network(embedded_strain)
        var  = self.var_network(embedded_strain) + self.eps
        self.mean = mean
        self.var  = var
        return torchMultivariateNormal(mean, scale_tril=torch.diag_embed(var)).log_prob(z_samples)
    
    def sample(self, num_samples, embedded_strain):
        #compute the mean and variance
        #NB here we assume embedded_strain has dim [1, strain_dim] (i.e. 1 sample per batch)
        mean = self.mean_network(embedded_strain).squeeze(0)
        var  = self.var_network(embedded_strain).squeeze(0) + self.eps
        self.mean = mean
        self.var  = var
        #by default .samples returns [1, num_samples, dim] so we delete 1 dimension
        return torchMultivariateNormal(mean, torch.diag_embed(var)).sample((1,num_samples)).squeeze(0)
    
# --------------------------------
# Resampled Multivariate Gaussian
# --------------------------------
class ResampledMultivariateNormalBase(nn.Module):
    """Resampled Multivariate Normal the base distribution for the flow. 
       The distribution is initialized to zero mean and unit variance. 
       Resampling is based on a Learned Acceptance Rejection Sampling (LARS) procedure,
       where each sample is accepted or rejected based on the output of a neural network. 
       
       Resampling procedure is described in arXiv:2110.15828. 
       The implementation follows https://github.com/VincentStimper/resampled-base-flows
    
        Args:
        -----
            dim (int)                  : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            T (int)                    : Maximum Number of rejections. (Default: 100)
            eps (float)                : Discount factor in exponential average of Z. (Default: 0.05)
            bs_factor (int)            : Factor to increment the batch size for the resampling. (Default: 1)
            acc_network_kwargs (dict)  : arguments to be passed to the neural network that conditions the distribution.
                
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
    def __init__(self, 
                 dim           :int   = 10,
                 T             :int   = 100,
                 eps           :float = 0.05,
                 bs_factor     :int   = 1,
                 trainable     :bool  = False,
                 acc_network_kwargs = {},
                 ):
        super(ResampledMultivariateNormalBase, self).__init__()
        
        self.dim = dim      
        self.T   = T
        self.eps = eps
        self.bs_factor = bs_factor
        self.register_buffer("Z", torch.tensor(-1.))
        
        #construct the acceptance network
        activation = acc_network_kwargs.get('activation', nn.ELU())
        dropout    = acc_network_kwargs.get('dropout', 0.2)
        layer_dim  = acc_network_kwargs.get('layer_dims', 256)
        
        self.acceptance_network = nn.Sequential(nn.LazyLinear(layer_dim), activation,
                                                nn.Linear(layer_dim, layer_dim), activation,
                                                nn.Dropout(dropout),
                                                nn.Linear(layer_dim, 1), nn.Sigmoid())
        
        self.multivariate_normal = MultivariateNormalBase(dim, trainable)
        return
    
    @property
    def mean(self):
        return self.multivariate_normal.mean.unsqueeze(0)
    
    @property
    def var(self):
        #REVIEW - check if this has to be the variance or the standard deviation
        var_ = self.multivariate_normal.var
        return torch.diagonal(var_, dim1=-2, dim2=-1)
    
    
    def log_prob(self, z_samples, embedded_strain=None):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong z_samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        #compute the acceptance probability on latent space samples
        z_eps = (z_samples - self.mean) / torch.exp(self.var)
        acceptance_prob = self.acceptance_network(z_eps)[:, 0] #[Nbatch, 1]->[Nbatch]
        
        #estimate Z with batch Monte Carlo
        z_ = torch.randn_like(z_samples)
        Z_batch = torch.mean(self.acceptance_network(z_))
        if self.Z < 0.:
            self.Z = Z_batch.detach()
        else:
            self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
        Z = Z_batch - Z_batch.detach() + self.Z
        
        #compute the new log_prob
        alpha = (1 - Z) ** (self.T - 1)
        log_prob_gaussian = self.multivariate_normal.log_prob(z_samples)
        return torch.log((1 - alpha) * acceptance_prob / Z + alpha) + log_prob_gaussian
    
    def sample(self, num_samples, embedded_strain=None):
        """Sample from the Multivariate Normal distribution using the 
              Learned Acceptance Rejection Sampling (LARS) procedure"""
        
        #initialize the samples
        samples = torch.empty((num_samples, self.dim), device = self.mean.device)
        
        it = 0
        it_max = self.T // self.bs_factor + 1
        t=0
        
        len_samples = 0
        
        #rejection sampling loop
        while it <= it_max:
            s_ = self.multivariate_normal.sample(num_samples, embedded_strain)
            acceptance_prob = self.acceptance_network((s_ - self.mean) / torch.exp(self.var))[:,0]
            
            accept = torch.rand_like(acceptance_prob) < acceptance_prob
            
            for isamp, a in enumerate(accept):
                #here either we accept the sample because of the acceptance 
                #probability or because we reached the maximum number of rejections
                if a or t == self.T-1:
                    samples[isamp] = s_[isamp]
                    len_samples += 1 #increment the number of samples counter
                    t = 0            #reset the rejection counter
                else:
                    #we reject the sample and increment the rejection counter
                    t += 1 
                if len_samples == num_samples:
                    break
            if len_samples == num_samples:
                break
        
        return samples
        
# --------------------------------------------
# Resampled Multivariate Gaussian Conditional
# --------------------------------------------
class ResampledConditionalMultivariateNormalBase(ResampledMultivariateNormalBase):
    """Resampled Multivariate Normal the base distribution for the flow conditioned on embedded context. 
       The distribution is initialized to zero mean and unit variance. 
       
       Resampling procedure is described in arXiv:2110.15828. 
       The implementation follows https://github.com/VincentStimper/resampled-base-flows
    
        Args:
        -----
            dim (int)                    : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            neural_network_kwargs (dict) : arguments to be passed to the neural network(s) that conditions the distribution.
                
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
    def __init__(self, 
                 dim           :int   = 10,
                 T             :int   = 100,
                 eps           :float = 0.05,
                 bs_factor     :int   = 1,
                 acc_network_kwargs   = {},
                 neural_network_kwargs= {},             
                 ):
        super().__init__(dim, T, eps, bs_factor, acc_network_kwargs)
        
        self.multivariate_normal = ConditionalMultivariateNormalBase(dim, neural_network_kwargs)
        return
    
    @property
    def mean(self):
        return self.multivariate_normal.mean
    
    @property
    def var(self):
        #REVIEW - check if this has to be the variance or the standard deviation
        var_ = self.multivariate_normal.var
        return torch.diagonal(var_, dim1=-2, dim2=-1)
    
    def log_prob(self, z_samples, embedded_strain=None):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong z_samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        #compute the gaussian log_prob
        #NOTE - we moved it here so that the means/vars are already computed by the ConditionalMultivariateNormalBase class
        log_prob_gaussian = self.multivariate_normal.log_prob(z_samples, embedded_strain)
        
        #compute the acceptance probability on latent space samples
        z_eps = (z_samples - self.mean) / torch.exp(self.var)
        acceptance_prob = self.acceptance_network(z_eps)[:, 0] #[Nbatch, 1]->[Nbatch]
        
        #estimate Z with batch Monte Carlo
        z_ = torch.randn_like(z_samples)
        Z_batch = torch.mean(self.acceptance_network(z_))
        if self.Z < 0.:
            self.Z = Z_batch.detach()
        else:
            self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
        Z = Z_batch - Z_batch.detach() + self.Z
        
        #compute the new log_prob
        alpha = (1 - Z) ** (self.T - 1)
        return torch.log((1 - alpha) * acceptance_prob / Z + alpha) + log_prob_gaussian
    
    def sample(self, num_samples, embedded_strain=None):
        """Sample from the Multivariate Normal distribution using the 
              Learned Acceptance Rejection Sampling (LARS) procedure"""
        
        #initialize the samples
        samples = torch.empty((num_samples, self.dim), device = self.mean.device)
        
        it = 0
        it_max = self.T // self.bs_factor + 1
        t=0
        
        len_samples = 0
        
        #rejection sampling loop
        while it <= it_max:
            s_ = self.multivariate_normal.sample(num_samples, embedded_strain)
            acceptance_prob = self.acceptance_network((s_ - self.mean) / torch.exp(self.var))[:,0]
            
            accept = torch.rand_like(acceptance_prob) < acceptance_prob
            
            for isamp, a in enumerate(accept):
                #here either we accept the sample because of the acceptance 
                #probability or because we reached the maximum number of rejections
                if a or t == self.T-1:
                    samples[isamp] = s_[isamp]
                    len_samples += 1 #increment the number of samples counter
                    t = 0            #reset the rejection counter
                else:
                    #we reject the sample and increment the rejection counter
                    t += 1 
                if len_samples == num_samples:
                    break
            if len_samples == num_samples:
                break
        
        return samples
    
    
    
# ------------------------------
# Multivariate Gaussian Mixture
# ------------------------------
class MultivariateGaussianMixtureBase(nn.Module):
    """Mixture of Multivariate Normal the base distribution for the flow. 
       Mixture weights are initialized to 1/N with N components.
    
        Args:
        -----
            dim (int)             : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            num_components (int)  : Number of components in the mixture. (Default: 2)
            trainable (bool)      : wether to learn the weights / means / stds of the mixture. (Default: True)
                
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_z, self.dim])
    
    """

    def __init__(self, 
                 dim            :int  =  10,
                 num_components :int  =  2,
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
    
    def log_prob(self, z_samples, embedded_strain=None):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        weights  = F.softmax(self.mixture_weights)

        log_prob = [weights[i].log() + self.mixture_components[i].log_prob(z_samples) for i in range(self.num_components)]
        
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

# ------------------------------------------
# Multivariate Gaussian Mixture Conditional
# ------------------------------------------
class ConditionalMultivariateGaussianMixtureBase(nn.Module):
    """Multivariate Normal the base distribution for the flow conditioned on embedded context. 
       Exploits the torch.distributions module
       The distribution is initialized to zero mean and unit variance. 
       Mixture weights are the output of a FC network conditioned on the context.
       Mixture components are instances of ConditionalMultivariateNormalBase
    
        Args:
        -----
            dim (int)                    : Number of dimensions (i.e. of physical inference parameters). (Default: 10)
            num_components (int)         : Number of components in the mixture. (Default: 2)
            neural_network_kwargs (dict) : arguments to be passed to the neural network(s) that conditions the distribution.
                
        Methods:
        --------
            - log_prob : returns the log_prob given samples of dim (N batches, self.dim)
            - sample   : samples from the prior distribution returning a tensor of dim [Num_samples, self.dim])
    
    """
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
                                                                                   neural_network_kwargs=neural_network_kwargs) 
                                                 for _ in range(num_components)])
        
        self.weights_network = nn.Sequential(nn.LazyLinear(layer_dim), activation,
                                             nn.Dropout(dropout),
                                             nn.Linear(layer_dim, layer_dim), activation,
                                             nn.Dropout(dropout),
                                             nn.Linear(layer_dim, num_components), nn.Softmax(dim=1))
    
        return
    
    
    
    def log_prob(self, z_samples, embedded_strain):
        """assumes that z_samples have dim (Nbatch, self.dim) ie 1 sample per batch"""
        
        if z_samples.shape[1] != self.dim:
            raise ValueError(f'Wrong z_samples dim. Expected (batch_size, {self.dim}) and got {z_samples.dim}')
        
        weights  = self.weights_network(embedded_strain)

        log_prob = [weights[:,i].log() + self.mixture_components[i].log_prob(z_samples, embedded_strain) for i in range(self.num_components)]
        
        return torch.sum(torch.stack(log_prob, axis=0), axis=0)
    
    def sample(self, num_samples, embedded_strain):
        #sample the component
        weights  = self.weights_network(embedded_strain)
        component_samples = torch.multinomial(weights, num_samples, replacement=True).squeeze(0)
        
        #sample from the components
        samples = torch.empty((num_samples, self.dim), device = embedded_strain.device)
        
        for i in range(self.num_components):
            mask = component_samples == i
            samples[mask, :] = self.mixture_components[i].sample(mask.sum(), embedded_strain)
            '''
            print(f'component: {i}')
            print(f'weight: {weights[:,i].item()}')
            print(f'mean: {self.mixture_components[i].mean_network(embedded_strain)}')
            print(f'var: {self.mixture_components[i].var_network(embedded_strain)}')
            '''

        return samples
    
# ---------------------------------
# Multivariate Gaussian & VonMises
# ---------------------------------
class VonMisesNormal(nn.Module):
    """It implements the base distribution for the flow. 
       Exploits the torch.distributions module
    
        Args:
        -----
            parameters (dict of float tuples) : dict containing tuples of paramters for the distributions. (Default: {'Normal': (0.0, 1.0), 'VonMises': (0.0, 1.0)})
            dim (dict of integers)            : dict specifying the dim of each distributions. (Default: {'Normal': 6, 'VonMises': 4} for the CE case)
            device (torch.device)             : used to specify the device (either 'cuda' or 'cpu'). (Default: 'cpu')  
            
        Methods:
        --------
            - log_prob  : returns the log_prob given samples of dim (N batches, Normal_dim + VonMises_dim, Num_samples)
            - sample    : samples from the prior distribution returning a tensor of dim (Normal_dim + VonMises_dim, Num_samples)
    
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
        
        
    def log_prob(self, z_samples, embedded_strain=None):
        """assumes that z_samples have dim (Nbatch, N_norm+N_vonmises) ie 1 sample per batch"""
        
        if z_samples.dim[1] != self.Normal_dim+self.VonMises_dim:
            raise ValueError("Wrong z_samples dim. Expected (None, %d, None) and got %s"%((self.Normal_dim+self.VonMises_dim), z_samples.dim))
        
        #print('>>>>>>>>>> z_samples dim', z_samples.dim)
        xn = z_samples[:, self.Normal_mask]
        xv = z_samples[:, self.VonMises_mask]
        
        
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
                           'ResampledMultivariateNormalBase'           : ResampledMultivariateNormalBase,
                           'ResampledConditionalMultivariateNormalBase': ResampledConditionalMultivariateNormalBase,
                           'MultivariateGaussianMixtureBase'           : MultivariateGaussianMixtureBase,
                           'ConditionalMultivariateGaussianMixtureBase': ConditionalMultivariateGaussianMixtureBase,
                           'VonMisesNormal'                            : VonMisesNormal}