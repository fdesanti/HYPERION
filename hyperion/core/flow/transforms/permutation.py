import torch
import torch.nn as nn


class RandomPermutation(nn.Module):
    """Implementation of Random Permutation transformation that shuffles parameters along the coupling transformation"""
    
    def __init__(self, num_features, kwargs = None):
        
        super(RandomPermutation, self).__init__()
        
        self.num_features = num_features
        
        self.register_buffer("_permutation", torch.randperm(num_features) )
        
        return
    
    def forward(self, inputs, embedded_strain=None):
        
        batch_size = inputs.shape[0]
        outputs    = inputs[:, self._permutation]
        logabsdet  = inputs.new_zeros(batch_size)
        
        return outputs, logabsdet
    
    def inverse(self, inputs, embedded_straub=None):
        
        batch_size = inputs.shape[0]
        outputs    = inputs[:, torch.argsort(self._permutation)]
        logabsdet  = inputs.new_zeros(batch_size)
        
        return outputs, logabsdet
