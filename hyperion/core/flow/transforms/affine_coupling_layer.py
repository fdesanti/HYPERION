"""Here it is implemented the AffineCouplingLayer for the flow. (See arXiv:1605.08803)"""

import torch
import torch.nn as nn

class AffineCouplingLayer(nn.Module):
    
    def __init__(self,
                 num_features        :int = 8,
                 strain_features     :int = 256,
                 num_identity        :int = 4,
                 num_transformed     :int = 4,
                 linear_dim          :int = 512,
                 dropout_probability :int = 0.2,
                 s_network           = None,
                 t_network           = None,
                 ):
        super(AffineCouplingLayer, self).__init__()
        
        self.num_features    = num_features
        self.num_identity    = num_identity
        self.num_transformed = num_transformed
        self.strain_features = strain_features
        
        assert num_features == num_identity + num_transformed, 'The number of features must be equal to the sum of the number of identity and transformed features'
    
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
    
        return

    
    def _coupling_transform(self, inputs, embedded_strain, inverse):
        #initialize the output
        outputs = torch.empty_like(inputs)                   #full of zeros of shape as input
        outputs[:, :self.num_identity] = inputs[:, :self.num_identity] #untransformed output
        

        x = torch.cat([inputs[:, :self.num_identity], embedded_strain], dim=1)
        s = self.s_network(x)
        
        t = self.t_network(x)
        if inverse:
            outputs[:, self.num_identity:] = (inputs[:, self.num_identity:] - t) * torch.exp(-s)
            logabsdet = -torch.sum(s, dim=(1))
            
        else:
            outputs[:, self.num_identity:] = inputs[:, self.num_identity:] * torch.exp(s) + t
            logabsdet = torch.sum(s, dim=(1))

        return outputs, logabsdet
    
    
    def forward(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
        
        
    def inverse(self, inputs, embedded_strain):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)
        
