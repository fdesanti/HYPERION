import torch
import torch.nn as nn

class RandomPermutation(nn.Module):
    """ 
    Randomly permutes the features of the input tensor.

    Args:
        num_features (int): Number of features in the input tensor.
        kwargs      (dict): Additional keyword arguments (unused).
    """
    
    def __init__(self, num_features, kwargs=None):
        super(RandomPermutation, self).__init__()
        
        self.num_features = num_features
        self.register_buffer("_permutation", torch.randperm(num_features) )
        

    def forward(self, inputs, embedded_strain=None):
        """
        Applies the random permutation to the input tensor.

        Args:
            inputs          (torch.Tensor): Input tensor.
            embedded_strain (torch.Tensor): Optional additional input (unused in this implementation).

        Returns:
            tuple: (outputs, logabsdet), where ``outputs`` is the permuted tensor and ``logabsdet`` is a tensor of zeros (no volume change in this permutation).
        """

        batch_size = inputs.shape[0]
        outputs    = inputs[:, self._permutation]
        logabsdet  = inputs.new_zeros(batch_size)
        
        return outputs, logabsdet
    
    def inverse(self, inputs, embedded_strain=None):
        """
        Reverses the random permutation applied to the input tensor.

        Args:
            inputs          (torch.Tensor): Input tensor.
            embedded_strain (torch.Tensor): Optional additional input (unused in this implementation).

        Returns:
            tuple: (outputs, logabsdet), where ``outputs`` is the original tensor before permutation and ``logabsdet`` is a tensor of zeros (no volume change).
        """
    
        batch_size = inputs.shape[0]
        outputs    = inputs[:, torch.argsort(self._permutation)]
        logabsdet  = inputs.new_zeros(batch_size)
        
        return outputs, logabsdet
    
class CyclicPermutation(nn.Module):
    """
    Cyclically shifts features along a specified dimension.

    Args:
        shift (int): Number of positions to shift. Positive values shift right, negative values shift left.
        dim   (int): The dimension along which to perform the cyclic shift.
    """
    def __init__(self, shift=1, dim=-1):
        super(CyclicPermutation, self).__init__()
        
        self.shift = shift
        self.dim = dim

    def forward(self, inputs, embedded_strain=None):
        """
        Applies the cyclic shift to the input tensor along the specified dimension.

        Args:
            inputs          (torch.Tensor): Input tensor.
            embedded_strain (torch.Tensor): Optional additional input (unused in this implementation).

        Returns:
            tuple: (outputs, logabsdet), where ``outputs`` is the permuted tensor and ``logabsdet`` is a tensor of zeros (no volume change in this permutation).
        """
        outputs = torch.roll(inputs, shifts=self.shift, dims=self.dim)
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def inverse(self, inputs, embedded_strain=None):
        """
        Reverses the cyclic shift applied to the input tensor.

        Args:
            inputs          (torch.Tensor): Input tensor.
            embedded_strain (torch.Tensor): Optional additional input (unused in this implementation).

        Returns:
            tuple: (outputs, logabsdet), where ``outputs`` is the original tensor before permutation and ``logabsdet`` is a tensor of zeros (no volume change).
        """
        outputs = torch.roll(inputs, shifts=-self.shift, dims=self.dim)
        batch_size = inputs.shape[0]
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

