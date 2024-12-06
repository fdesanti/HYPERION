"""Here it is implemented the AffineCouplingLayer for the flow. (See arXiv:1605.08803)"""

import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    """
        Affine Coupling Layer implementation for the flow.
        The forward transformation is defined as an exponential rescaling plus a translation:
            y = exp(s(x)) * x + t(x)
        
        The inverse transformation is defined as:
            x = (y - t(x)) * exp(-s(x))

        If volume_preserving is set to True then we constrain the Jacobian to be 1 by setting 
            s -> s - s.mean()
        This ensure that volume is preserved by the transformation (i.e. incompressible flow)
        
        Args:
        -----
            num_features              (int): Number of features in the input tensor
            num_identity              (int): Number of features that are not transformed. 
                                             If None then num_features - num_features//2 (Default. None)
            num_transformed           (int): Number of features that are transformed. 
                                             If None then num_features//2 (Default. None)
            linear_dim                (int): Dimension of the linear layers
            s_network           (nn.Module): Network that computes the scale factor. (Default. None)
            t_network           (nn.Module): Network that computes the translation factor. (Default. None)
            dropout_probability     (float): Dropout probability (Default. 0.2)
            volume_preserving        (bool): If True, the Jacobian is constrained to be 1. (Default. False)

        Note:
        -----
            The number of features must be equal to the sum of the number of identity and transformed features. 
            If the networks are not provided, they are initialized as in (Phys. Rev. D 109, 102004)
    """
    def __init__(self,
                 num_features,
                 num_identity        = None,
                 num_transformed     = None,
                 linear_dim          = 512,
                 s_network           = None,
                 t_network           = None,
                 dropout_probability = 0.2,
                 volume_preserving   = False
                 ):
        super(AffineCouplingLayer, self).__init__()
        
        self.num_features      = num_features
        self.num_identity      = num_identity    if num_identity is not None else num_features - num_features//2
        self.num_transformed   = num_transformed if num_transformed is not None else num_features//2
        self.volume_preserving = volume_preserving
        
        assert self.num_features == self.num_identity + self.num_transformed, 'The number of features must be equal to the sum of the number of identity and transformed features'
    
        if s_network is not None:
            self.s_network = s_network
        else:
            s_activation = nn.Tanh()
            self.s_network = nn.Sequential(nn.LazyLinear(linear_dim), s_activation, 
                                           nn.LazyLinear(linear_dim), s_activation, 
                                           nn.LazyLinear(linear_dim), s_activation, 
                                           nn.Dropout(dropout_probability), 
                                           nn.LazyLinear(num_transformed), s_activation)
        if t_network is not None:
            self.t_network = t_network
        else:
            t_activation = nn.ELU()
            self.t_network = nn.Sequential(nn.LazyLinear(linear_dim), t_activation, 
                                           nn.LazyLinear(linear_dim), t_activation, 
                                           nn.LazyLinear(linear_dim), t_activation, 
                                           nn.Dropout(dropout_probability), 
                                           nn.LazyLinear(num_transformed), t_activation)
            

    def _coupling_transform(self, inputs, embedded_strain, inverse):
        #initialize the output
        outputs = torch.empty_like(inputs)                             #full of zeros of shape as input
        outputs[:, :self.num_identity] = inputs[:, :self.num_identity] #untransformed output
        
        if embedded_strain is not None:
            x = torch.cat([inputs[:, :self.num_identity], embedded_strain], dim=1)
        else:
            x = inputs[:, :self.num_identity]
        
        #compute the scale
        s = self.s_network(x)
        if self.volume_preserving:
            s = s - s.mean(-1, keepdim=True) #assumes shape (batch, num_transformed)

        #compute the translation
        t = self.t_network(x)

        #coupling transformation
        if inverse:
            outputs[:, self.num_identity:] = (inputs[:, self.num_identity:] - t) * torch.exp(-s)
            logabsdet = -torch.sum(s, dim=(1))
        else:
            outputs[:, self.num_identity:] = inputs[:, self.num_identity:] * torch.exp(s) + t
            logabsdet = torch.sum(s, dim=(1))

        return outputs, logabsdet
    
    
    def forward(self, inputs, embedded_strain=None):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
        
    def inverse(self, inputs, embedded_strain=None):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)
        
