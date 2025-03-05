""" Here it is implemented the Coupling transform for the flow."""

import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    """
    Base class for the CouplingLayer
    
    Args:
        num_features     (int): Number of features in the input tensor
        num_identity     (int): Number of features that are not transformed. If None then ``num_features - num_features//2`` (Default: None)
        num_transformed  (int): Number of features that are transformed. If None then ``num_features//2`` (Default: None)
    """
    def __init__(self, 
                 num_features, 
                 num_identity    = None,
                 num_transformed = None,
                 ):
        super(CouplingLayer, self).__init__()
        self.num_features    = num_features
        self.num_identity    = num_identity    if num_identity    is not None else num_features - num_features // 2
        self.num_transformed = num_transformed if num_transformed is not None else num_features // 2
        assert self.num_features == self.num_identity + self.num_transformed, 'The number of features must be equal to the sum of the number of identity and transformed features'

    def _coupling_transform(self, inputs, embedded_strain, inverse):
        raise NotImplementedError

    def forward(self, inputs, embedded_strain=None):
        return self._coupling_transform(inputs, embedded_strain, inverse=False)
    
    def inverse(self, inputs, embedded_strain=None):
        return self._coupling_transform(inputs, embedded_strain, inverse=True)

class CouplingTransform(nn.Module):
    """
    Class that implements the full Coupling transform
    
    Args:
         transform_layers (list): List of AffineCouplingLayers instances
    """
    
    def __init__(self, transform_layers):
        super().__init__()
        
        self._transforms = nn.ModuleList(transform_layers)        
        return
    
    @staticmethod
    def _cascade(inputs, layers, embedded_strain=None):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for layer in layers:
            outputs, logabsdet = layer(outputs, embedded_strain)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs, embedded_strain=None):
        r"""
        Forward pass (training)

        .. math:: 
        
            \theta  \rightarrow z = f_\phi(\theta, s)
        
        Args: 
            inputs          (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            embedded_strain (torch.Tensor): (Optional) Embedded strain tensor of shape [N, E] where N is the number of samples and E is the dimension of the embedding.
        """
        layers = self._transforms
        return self._cascade(inputs, layers, embedded_strain)

    def inverse(self, inputs, embedded_strain=None):
        r"""
        Inverse pass (inference)

        .. math::
        
            z \rightarrow \theta = f_\phi^{-1}(z, s)

        Args:
            inputs          (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            embedded_strain (torch.Tensor): (Optional) Embedded strain tensor of shape [N, E] where N is the number of samples and E is the dimension of the embedding.
        """
        layers = (transform.inverse for transform in self._transforms[::-1])
        if embedded_strain is not None and inputs.shape[0]!= embedded_strain.shape[0]:
            s = torch.cat([embedded_strain for _ in range(inputs.shape[0])], dim = 0)
        else:
            s = embedded_strain
        return self._cascade(inputs, layers, s)