""" Here it is implemented the Coupling transform for the flow."""

import torch
import torch.nn as nn

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