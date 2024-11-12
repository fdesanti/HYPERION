try:
    from .kenn_embedding import *
except ImportError as e:
    print(f"[ERROR]: Cannot import KennEmbedding: {e}")
    print("[WARNING]: Please install the deepfastet package to use the KennEmbedding.")
    KennEmbedding = None
    pass

from .strain_embedding import *
from .strain_embedding_attention import *


embedding_network_dict = {'KENN': KennEmbedding,
                          'CNN+ResNet': EmbeddingNetwork, 
                          'CNN+ResNet+Attention': EmbeddingNetworkAttention
                          }