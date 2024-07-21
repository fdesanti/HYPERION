from .strain_embedding import *
from .strain_embedding_attention import *


embedding_network_dict = {'CNN+ResNet': EmbeddingNetwork, 
                          'CNN+ResNet+Attention': EmbeddingNetworkAttention}