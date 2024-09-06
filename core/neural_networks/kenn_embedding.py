"""Wrapper for the KENN embedding."""

from deepfastet.model import Kenn, KennConfig, KennAttention, KennAttentionHead

class KennEmbedding(Kenn):
    """Wrapper for the KENN embedding."""
    
    def __init__(self, config_kwargs):
        
        kenn_config = KennConfig(KennAttention, 
                                 KennAttentionHead, 
                                 **config_kwargs)
        
        super().__init__(kenn_config)
    

