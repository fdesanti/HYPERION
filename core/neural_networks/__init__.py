from .kenn_embedding import *
from .strain_embedding import *
from .transformer_embedding import *
from .strain_embedding_attention import *


embedding_network_dict = {'KENN': KennEmbedding,
                          'CNN+ResNet': EmbeddingNetwork, 
                          'Transformer': TransformerEmbedding,
                          'CNN+ResNet+Attention': EmbeddingNetworkAttention
                          }