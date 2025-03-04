"""Wrapper for the KENN embedding."""
import inspect
import torch.nn as nn
from deepfastet.model import Kenn, KennConfig, KennAttention, KennAttentionHead
from .torch_layers import ResBlock

class KennEmbedding(Kenn):
    """
    Wrapper for the KENN embedding.

    Args:
        strain_shape  (list): Shape of the input strain tensor
        fs            (int): Sampling frequency of the strain data
        kwargs: Additional keyword arguments to initialize the KENN network
    """
    
    def __init__(self, strain_shape, fs, **kwargs):

        #n_channels, _ = strain_shape
        
        #default config values
        config = {'d_model'       : 512,
                  'duration_in_s' : 2,  # s
                  'sampling_rate' : fs,
                  'chunk_size'    : 0.5,
                  'n_heads'       : 32,
                  'dropout'       : 0.4,
                  'encoder_layers': 2,
                  'n_channels'    : 3, 
                }
        #update with user
        config.update(kwargs)
        #extract only the parameters accepted by KennConfig
        kenn_params = self._get_kenn_params(config)
        
        #create a KennConfig instance
        kenn_config = KennConfig(KennAttentionHead, KennAttention, **kenn_params)
        super().__init__(kenn_config)
        '''
        #=======================================================================
        # Construct ResNet blocks
        #=======================================================================
        block_dims = kwargs.get('block_dims', [64, 128, 256])
        num_blocks = kwargs.get('num_blocks', 3)
        activation = kwargs.get('activation', nn.ReLU())
        dropout_probability = kwargs.get('dropout_probability', 0.2)

        self.residual_blocks = []
        input_dim = block_dims[0]
        for block_dim in block_dims:
            for _ in range(num_blocks):
                self.residual_blocks.append( ResBlock(input_dim           = input_dim, 
                                                      output_dim          = block_dim,
                                                      use_batch_norm      = False,
                                                      activation          = activation,
                                                      dropout_probability = dropout_probability) )
                input_dim = block_dim
            
        self.resnet = nn.ModuleList(self.residual_blocks)
        
        #pre ResNet Layer
        self.pre_resnet_linear = nn.Sequential(nn.LazyLinear(block_dims[0]), activation)
        '''
    def _get_kenn_params(self, config):
        # Obtains the name of the parameters accepted by KennConfig
        kenn_init_params = inspect.signature(KennConfig).parameters
        return {k: v for k, v in config.items() if k in kenn_init_params}
    '''
    def forward(self, strain, asd=None):
        #We first apply the KENN embedding forward method and then the ResNet blocks
        embedded_strain = super().forward(strain, asd)

        #apply the pre-resnet linear layer
        embedded_strain = self.pre_resnet_linear(embedded_strain)

        #Final Resnet
        for res_layer in self.resnet:
            embedded_strain = res_layer(embedded_strain)        
        
        return embedded_strain
    '''