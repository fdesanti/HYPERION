""" Here it is implemented the Coupling transform for the flow."""

import torch
import torch.nn as nn


#torch.manual_seed(0)

class CouplingTransform(nn.Module):
    """Class that implements the full Coupling transform
    
    Args:
    ----
         transform_layers : list 
            List of AffineCouplingLayers instances
        
    Methods:
    --------
        - forward: it's used during training  to map x --> u, where u = prior(u)
        - inverse: it's used during inference to map u --> x, where u = prior(u)
    """
    
    def __init__(self, transform_layers):
        super().__init__()
        
        self._transforms = nn.ModuleList(transform_layers)        
        return
    
    @staticmethod
    def _cascade(inputs, layers, embedded_strain):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for layer in layers:
            outputs, logabsdet = layer(outputs, embedded_strain)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, embedded_strain):
        """Forward pass (training)
        Args: 
            - inputs: tensor of shape [N batch, N posterior parameters] coming from the dataset
            - embedded_strain: tensor of shape [N batch, 3, len_embedded_strain] embedded_strain associated to each element of the dataset
                      to contrain the flow
        Reuturns:
            - _cascade() of the layers in self._transforms 
        """
        layers = self._transforms
        return self._cascade(inputs, layers, embedded_strain)

    def inverse(self, inputs, embedded_strain):
        """Inverse pass (inference)
        Args: 
            - inputs: tensor of shape [N prior samples, N posterior parameters] coming from the sampled prior
            - embedded_strain: tensor of shape [1, 3, len_embedded_strain] embedded_strain associated to which predict the posterior
            
        Reuturns:
            - _cascade() of the layers in self._transforms.inverse in reversed order 
        """
        layers = (transform.inverse for transform in self._transforms[::-1])
        if inputs.shape[0]!= embedded_strain.shape[0]:
            s = torch.cat([embedded_strain for _ in range(inputs.shape[0])], dim = 0)
        else:
            s = embedded_strain
        return self._cascade(inputs, layers, s)

    
    
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
        
        
    

















################################################
################ TESTING #######################
################################################ 
if __name__ == '__main__':
    from neural_network.network_architecture import ConvResNet

    print('Testing...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size, n_det, len_embedded_strain = 256, 3, 4096
    embedded_strain_input_shape = (batch_size, n_det, len_embedded_strain)

    embedded_strain               = torch.rand(batch_size, n_det, len_embedded_strain).to(device)
    untransformed_params = torch.rand(batch_size, 5).to(device)

    #model = ConvBlock(input_shape, use_separable_conv=False, pooling='max')
    model = ConvResNet((n_det, len_embedded_strain), num_bins= 10, transformed_params_size=5)
    model = model.to(device)
    print('ok resnet')
    
    
    coupling = SplineCouplingLayer(neural_network = model)
    
    print('ok coupling')
    
    new_mask = coupling._permute_mask(coupling.transform_mask)
    print(new_mask)

