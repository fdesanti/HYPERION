import torch
import torch.nn as nn
from torch.distributions import Normal as torchNormal
from torch.distributions import VonMises as torchVonMises
from torch.distributions import MultivariateNormal as torchMultivariateNormal

class MultivariateNormalPrior(nn.Module):
    def __init__(self, 
                 shape      :int  =  8,
                 trainable  :bool = False,
                 ):
        super(MultivariateNormalPrior, self).__init__()
        self.shape = shape
        self.trainable = trainable
        
        self.initialize_distribution()
        return
    
    def initialize_distribution(self):
        """Initializes the distributions given the paramters from the __init__"""
        
        if self.trainable:
            self.mean = nn.Parameter(torch.zeros(self.shape))
            self.var  = nn.Parameter(torch.eye(self.shape))
        else:
            self.register_buffer("mean", torch.zeros(self.shape))
            self.register_buffer("var",  torch.eye(self.shape))

    @property
    def MultivariateNormal(self):
        return torchMultivariateNormal(self.mean, self.var, validate_args = False)
    
    def log_prob(self, samples):
        """assumes that samples have shape (Nbatch, self.shape) ie 1 sample per batch"""
        
        if samples.shape[1] != self.shape:
            raise ValueError(f'Wrong samples shape. Expected (batch_size, {self.shape}) and got {samples.shape}')
        
    
        return self.MultivariateNormal.log_prob(samples)
    
    def sample(self, num_samples):
        
        #by default .samples returns [1, num_samples, shape] so we delete 1 dimension
        
        return self.MultivariateNormal.sample((1, num_samples)).squeeze(0)

    


class VonMisesNormal(nn.Module):
    """It implements the base distribution for the flow. 
       Exploits the torch.distributions module
    
        Args:
            - parameters (dict of tuples): dict containing tuples of paramters for the distributions. These needs to be float
                                          (default {'Normal': (0.0, 1.0), 'VonMises': (0.0, 1.0)})
            - shape (dict of integers):  dict specifying the shape of each distributions.
                                        (default: {'Normal': 6, 'VonMises': 4} for the CE case)
            
            - device (torch.device): used to specify the device (cuda or cpu)  
            
        Methods:
            - log_prob: returns the log_prob given samples of shape (N batches, Normal_shape + VonMises_shape, Num_samples)
            
            - sample: samples from the prior distribution returning a tensor of shape (Normal_shape + VonMises_shape, Num_samples)
    
    """
    
    def __init__(self, 
                 parameters = {'Normal': [0.0, 1.0], 'VonMises': [0.0, 1.0]},
                 shape      = {'Normal': 8, 'VonMises': 0},
                 trainable  = False,
                 ):
        super(VonMisesNormal, self).__init__()

        #ASSERTION CHECKS ---------------------------------------------
        assert (parameters is not None) and shape is not None, 'Must specify <parameters> and <shape>'
        assert len(parameters) == len(shape), 'Number of parameters and dim does not match: found %d distributions and %d shape'%(len(parameters), len(shape))
        assert len(shape) == 2, 'I can only accept 2 distribution instances'
        
        if not all(isinstance(parameters[item], list) for item in parameters):
            raise ValueError('Expected <parameters> to be a list of tuples, got', parameters)
        
        if not all(isinstance(shape[item], int) for item in shape):
            raise ValueError('Expected <shape> to be a list of integers')
        #----------------------------------------------------------------

        
        self.parameters     = parameters
        self.Normal_shape   = shape['Normal']
        self.VonMises_shape = shape['VonMises']
        
        #DISTRIBUTION INITIALIZATION --------------------------------
        self.initialize_distributions(trainable)
        
        self.Normal_mask, self.VonMises_mask = self._create_Normal_and_VonMises_masks()
        return
    
    
    def initialize_distributions(self, trainable):
        """Initializes the distributions given the paramters from the __init__"""
        #Normal
        if trainable:
            self.norm_mean = nn.Parameter(torch.zeros(self.Normal_shape))
            self.norm_var  = nn.Parameter(torch.ones(self.Normal_shape))
        else:
            self.register_buffer("norm_mean", torch.tensor(self.parameters['Normal'][0]))
            self.register_buffer("norm_var",  torch.tensor(self.parameters['Normal'][1]))

        #VonMises
        if trainable:
            self.von_mises_mean = nn.Parameter(torch.zeros(self.VonMises_shape))
            self.concentration  = nn.Parameter(torch.ones(self.VonMises_shape))
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
        true  = torch.ones(self.Normal_shape, dtype = torch.bool)
        false = torch.zeros(self.VonMises_shape, dtype = torch.bool)
        Normal_mask = torch.cat([true, false], axis = 0)
        VonMises_mask = ~Normal_mask
        return Normal_mask, VonMises_mask
        
        
    def log_prob(self, samples):
        """assumes that samples have shape (Nbatch, N_norm+N_vonmises) ie 1 sample per batch"""
        
        if samples.shape[1] != self.Normal_shape+self.VonMises_shape:
            raise ValueError("Wrong samples shape. Expected (None, %d, None) and got %s"%((self.Normal_shape+self.VonMises_shape), samples.shape))
        
        #print('>>>>>>>>>> samples shape', samples.shape)
        xn = samples[:, self.Normal_mask]
        xv = samples[:, self.VonMises_mask]
        
        
        log_prob = [ self.Normal.log_prob(xn), self.VonMises.log_prob(xv) ]   
        
        log_prob = torch.cat(log_prob, axis = 1 )
        
                
        return torch.sum(log_prob, dim=(1))
    
    
    def sample(self, num_samples):
        
        #Norm_samples      = torch.fmod(self.Normal.sample((self.Normal_shape , num_samples)), 2).T
        #VonMises_samples  = torch.fmod(self.VonMises.sample((self.VonMises_shape,  num_samples)), 3).T

        Norm_samples      = self.Normal.sample((self.Normal_shape, num_samples)).T
        VonMises_samples  = self.VonMises.sample((self.VonMises_shape, num_samples)).T

        
        samples = torch.empty((num_samples, self.Normal_shape+self.VonMises_shape), device = self.norm_mean.device) #has shape [num samples, n CE parameters ]
        
        samples[:, self.Normal_mask]   = Norm_samples
        samples[:, self.VonMises_mask] = VonMises_samples
        
        return samples
        
        
        
        
        
    
    
    
        

   
#######################################
#============== TEST =================#
#######################################
def main_test():
    from tqdm import tqdm
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device %s'%device)
    
    #prior = VonMisesNormal(parameters={'Normal':(0.0, 1.1), 'VonMises':(0.0, 1.1)}, shape = {'Normal':6, 'VonMises':4}, device=device)
    prior = VonMisesNormal(device=device)
    
    for _ in tqdm(range(10)):
        x = prior.sample(num_samples = 10000)
  
    '''
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(len(x)):
        plt.subplot(len(x), 1, i+1)
        plt.hist(x[i], 100)
    plt.show()
    '''
    
    #print(x, x.shape)
    print(x.device)
    
    x = torch.unsqueeze(x, 0)
    print(x.shape)
    log_prob=prior.log_prob(x)
    print(log_prob)
   
    #print(x.T.shape)
    
    '''
    import matplotlib.pyplot as plt
    x = x.T
    for i in range(10):
    
        plt.hist(x[:, i], 100)
        plt.title(str(i))
        plt.show()
    '''
    x = torch.cat([x, x, x, x], axis = 0)
    log_prob=prior.log_prob(x)
    print(log_prob)
    
    return

if __name__ == '__main__':
    main_test()
    
    
    
