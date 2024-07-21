from .strain_embedding import EmbeddingNetwork
from .strain_embedding_attention import EmbeddingNetworkAttention


embedding_network_dict = {'EmbeddingNetwork': EmbeddingNetwork, 
                          'EmbeddingNetworkAttention': EmbeddingNetworkAttention}