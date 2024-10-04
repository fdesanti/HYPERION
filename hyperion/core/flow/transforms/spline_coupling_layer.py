"""Here it is implemented the SplineCouplingLayer for the flow. (See arXiv:1906.04032)"""


import torch
import torch.nn as nn

from .splines import unconstrained_rational_quadratic_spline

class SplineCouplingLayer(nn.Module):
    """
    Class that implements the single Coupling Layer with RationalQuadraticSplines
    
    Args:
        - input_dim (int):  dimensionality of the parameter space
        - num_circular (int): number of angles in parameters (ie. number of elements to transform with circular splines)
        - neural_network (nn.Module):  the neural network that produces the knots and is trained with the flow
        - layer_index (int):  index of the layer in the whole coupling transform. Used to permute the masks accordingly   
    
    Methods:
        - _create_transform_mask:  produces a 1D binary tensor mask with the first num_identity = False and the rest num_transformed = True
        - _create_circular_mask:   produces a 1D binary tensor mask with the first num_non_circuar = False and the rest num_circular = True
        - _permute_mask:           permutes cyclically the given mask 
        
        - _coupling_transform :    implements the coupling transform with splines 
        - forward :                makes the forward pass (training)
        - inverse :                makes the inverse pass (training)
    
    """
    
    def __init__(self,
                 num_features        = 10,
                 num_identity        =  5,
                 num_transformed     =  5,
                 num_bins            =  8,
                 tail_bound          =  2,
                 linear_dim          =  512,
                 activation          =  nn.ELU(),
                 dropout_probability = 0.2
                 ):
        super(SplineCouplingLayer, self).__init__()
        
        #assert isinstance(neural_network, nn.Module), 'A torch neural network module must be passed'
        
    
        self.num_features       = num_features
        self.num_identity       = num_identity
        self.num_transformed    = num_transformed
        self.tail_bound         = tail_bound
        self.num_bins           = num_bins
        self.linear_dim         = linear_dim
        
    
        self.h_network = nn.Sequential(nn.LazyLinear(linear_dim), activation, 
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(num_transformed*num_bins))
        
        self.w_network = nn.Sequential(nn.LazyLinear(linear_dim), activation,
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(num_transformed*num_bins))
        
        self.d_network = nn.Sequential(nn.LazyLinear(linear_dim), activation,
                                       nn.Dropout(dropout_probability), 
                                       nn.LazyLinear(num_transformed*(num_bins-1)))
        
        self.scaling_factor = torch.sqrt(torch.tensor(self.linear_dim).float())
    
        assert num_features == num_identity + num_transformed, 'The number of features must be equal to the sum of the number of identity and transformed features'
                                                                                                                                                     

        
        return
    

    
    def _coupling_transform(self, inputs, embedded_strain, inverse=False):
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
        x = torch.cat([inputs[:, :self.num_identity], embedded_strain], dim=1)
        
    
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
    


    def forward(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
        
    def inverse(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)

