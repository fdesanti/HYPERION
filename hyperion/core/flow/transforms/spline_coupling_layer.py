"""Here it is implemented the SplineCouplingLayer for the flow. (See arXiv:1906.04032)"""


import torch
import torch.nn as nn

from .splines import unconstrained_rational_quadratic_spline
from .coupling_transform import CouplingLayer

class SplineCouplingLayer(CouplingLayer):
    """
    Class that implements the single Coupling Layer with RationalQuadraticSplines. 
    (See arXiv:1906.04032)
    
    Args:
        num_features          (int): Number of features in the input tensor
        num_identity          (int): Number of features that are not transformed. If None then ``num_features - num_features//2`` (Default. None)
        num_transformed       (int): Number of features that are transformed. If None then ``num_features//2`` (Default. None)
        num_bins              (int): Number of bins in the spline (Default. 8)
        tail_bound          (float): Tail bound of the spline (Default. 2.0)
        linear_dim            (int): Dimension of the linear layers (Default. 512)
        activation      (nn.Module): Activation function (Default. nn.ELU())
        dropout_probability (float): Dropout probability (Default. 0.2)
    """
    def __init__(self,
                 num_features,
                 num_identity        =  None,
                 num_transformed     =  None,
                 num_bins            =  8,
                 tail_bound          =  2.0,
                 linear_dim          =  512,
                 activation          =  nn.ELU(),
                 dropout_probability = 0.2
                 ):
        super(SplineCouplingLayer, self).__init__(num_features, num_identity, num_transformed)
        
        self.tail_bound         = tail_bound
        self.num_bins           = num_bins
        self.linear_dim         = linear_dim
        
        self.h_network = nn.Sequential(nn.LazyLinear(linear_dim), activation, 
                                       nn.LazyLinear(linear_dim), activation,
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(self.num_transformed*num_bins))
        
        self.w_network = nn.Sequential(nn.LazyLinear(linear_dim), activation,
                                       nn.LazyLinear(linear_dim), activation,
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(self.num_transformed*num_bins))
        
        self.d_network = nn.Sequential(nn.LazyLinear(linear_dim), activation,
                                       nn.LazyLinear(linear_dim), activation,
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(self.num_transformed*(num_bins-1)))
        
        self.scaling_factor = torch.sqrt(torch.tensor(self.linear_dim).float())

    
    def _coupling_transform(self, inputs, embedded_strain=None, inverse=False):
        """Implements the coupling transform in both forward/inverse pass
        
        Args:
            if inverse = False <---- forward pass (training):
                inputs : tensor of shape (N batch, N posterior parameters) coming from the training dataset
                embedded_strain : tensor of shape (N batch, 3, Length embedded_strain) embedded_strain coming from the training dataset
                
            if inverse = True <----- inverse pass (inference):
                inputs : tensor of shape (N prior samples, N posterior parameters) tensor of prior samples
                embedded_strain : tensor of shape (1, 3, Length embedded_strain) single embedded_strain to analyze
                
                
        Returns:
            - outputs: tensor of same shape as input of parameters mapped by the coupling layer
            - logabsdet: tensor of shape [input.shape[0], 1] in both forward/inverse mode

        """
        batch_size = inputs.shape[0]
         
        #initialize the output
        outputs = torch.empty_like(inputs)                   #full of zeros of shape as input
        outputs[:, :self.num_identity] = inputs[:, :self.num_identity] #untransformed output
        
        #concatenate the identity inputs with the embedded strain
        if embedded_strain is not None:
            x = torch.cat([inputs[:, :self.num_identity], embedded_strain], dim=1)
        else:
            x = inputs[:, :self.num_identity]
        
    
        #compute the widths, heights and derivatives and reshape them into the right shape
        widths      = self.w_network(x).view(batch_size, self.num_transformed, self.num_bins)   #has shape (Nbatch, num_transformed, num_bins)
        heights     = self.h_network(x).view(batch_size, self.num_transformed, self.num_bins)   #has shape (Nbatch, num_transformed, num_bins)
        derivatives = self.d_network(x).view(batch_size, self.num_transformed, self.num_bins-1) #has shape (Nbatch, num_transformed, num_bins-1)
        
        #rescale the widths and heights
        widths  /= self.scaling_factor
        heights /= self.scaling_factor
        
        #transform the inputs with the splines
        transformed, logabsdet = unconstrained_rational_quadratic_spline(inputs[:, self.num_identity:], 
                                                                          widths,
                                                                          heights,
                                                                          derivatives,
                                                                          inverse    = inverse,
                                                                          tails      = 'linear',
                                                                          tail_bound = self.tail_bound)
        outputs[:, self.num_identity:] = transformed
            
        return outputs, logabsdet.sum(dim=1)