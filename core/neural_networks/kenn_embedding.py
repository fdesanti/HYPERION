"""Wrapper for the KENN embedding."""

from deepfastet.model import Kenn, KennConfig, KennAttention, KennAttentionHead

class KennEmbedding(Kenn):
    """Wrapper for the KENN embedding."""
    
    def __init__(self, strain_shape, fs, **kwargs):

        n_channels, _ = strain_shape
        
        #default config values
        config = {'d_model'       : 512,
                  'duration_in_s' : 32,  # s
                  'sampling_rate' : fs,
                  'chunk_size'    : 0.2,
                  'n_heads'       : 32,
                  'dropout'       : 0.4,
                  'encoder_layers': 2,
                  'n_channels'    : n_channels, 
                }
        #update with user
        config.update(kwargs)

        #create a KennConfig instance
        kenn_config = KennConfig(KennAttention, 
                                 KennAttentionHead, 
                                 **config)
        
        super().__init__(kenn_config)
    

