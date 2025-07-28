""" Here it is implemented the Coupling transform for the flow."""

import torch
import torch.nn as nn

from .permutation import Permutation

class CouplingLayer(nn.Module):
    r"""
    Base class for the CouplingLayer. The forward and inverse transformations must be implemented in the derived classes.
        
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

    def _coupling_transform(self, inputs, embedded_strain, inverse, input_mask):
        raise NotImplementedError

    def forward(self, inputs, embedded_strain=None, input_mask=None):
        """
        Computes the forward pass of the CouplingLayer

        Args: 
            inputs          (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            embedded_strain (torch.Tensor): (Optional) Embedded strain tensor of shape [N, E] where N is the number of samples and E is the dimension of the embedding.
        """
        return self._coupling_transform(inputs, embedded_strain, inverse=False, input_mask=input_mask)
    
    def inverse(self, inputs, embedded_strain=None, input_mask=None):
        """
        Computes the inverse pass of the CouplingLayer

        Args:
            inputs          (torch.Tensor): Tensor of shape [N, P] where N is the number of samples and P is the number of parameters
            embedded_strain (torch.Tensor): (Optional) Embedded strain tensor of shape [N, E] where N is the number of samples and E is the dimension of the embedding.
        """
        return self._coupling_transform(inputs, embedded_strain, inverse=True, input_mask=input_mask)

class CouplingTransform(nn.Module):
    """
    Class that implements the full Coupling transform
    
    Args:
         transform_layers (list): List of AffineCouplingLayers instances
    """
    
    def __init__(self, transform_layers, input_mask=None):
        super().__init__()
        
        self._transforms = nn.ModuleList(transform_layers)    
        self._input_mask = input_mask
    
    @property
    def input_mask(self):
        """
        Returns the input mask if it exists, otherwise returns None.
        """
        return self._input_mask

    @staticmethod
    def _cascade(inputs, layers, embedded_strain=None, input_mask=None):
        batch_size, num_features = inputs.shape
        
        total_logabsdet = inputs.new_zeros(batch_size)

        if input_mask is not None:
            input_mask = input_mask.unsqueeze(0).expand(batch_size, num_features).float()
        else:
            input_mask = torch.ones_like(inputs, dtype=torch.float)

        outputs = inputs

        for layer in layers:
            if isinstance(layer, Permutation) and input_mask is not None:
                input_mask, _ = layer(input_mask)
            outputs, logabsdet = layer(outputs, embedded_strain, input_mask=input_mask)
    
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
        return self._cascade(inputs, layers, embedded_strain, input_mask=self._input_mask)

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
        return self._cascade(inputs, layers, s, input_mask=self._input_mask)