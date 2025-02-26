import torch
from tensordict import TensorDict

#=======================================
#TensorDict Wrapper Class
#=======================================
class TensorSamples(TensorDict):
    """Wrapper class for TensorDict to better manage samples"""
    
    def flatten(self):
        """Returns the samples as a single tensor"""
        return torch.stack([self[key] for key in self.keys()], dim=-1)
    
    def to_tensor(self):
        """Returns the samples as a single tensor"""
        return self.flatten()
    
    def numpy(self):
        """Returns the samples as a numpy array"""
        return self.flatten().cpu().numpy()
