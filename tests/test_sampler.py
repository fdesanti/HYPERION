"""Test script for flow module."""

import pytest
import torch

from hyperion import PosteriorSampler
from hyperion.core.flow import *
from hyperion.core.distributions import *
from hyperion.core.neural_networks import EmbeddingNetworkAttention


def test_sampler():
    """Test unconditional flow training."""

    #set the prior
    prior = MultivariatePrior(dict(x=UniformPrior(-1, 1), y=UniformPrior(-1, 1)))

    #set the embedding_net
    channels = 3
    duration = 2
    fs = 2048

    embedding_net = EmbeddingNetworkAttention(strain_shape=[channels, duration*fs], fs=fs)

    #build the flow
    base_dist = MultivariateNormalBase(dim=2)
    coupling_layers = []
    for _ in range(4):
        coupling_layers.append(AffineCouplingLayer(num_features=2))
        coupling_layers.append(RandomPermutation(num_features=2))

    flow = Flow(base_distribution = base_dist, 
                embedding_network = embedding_net,
                transformation    = CouplingTransform(coupling_layers),
                metadata          = dict(prior_metadata=prior.metadata))
    
    sampler = PosteriorSampler(flow=flow)

    data = torch.randn((1, channels, duration*fs))
    posterior = sampler.sample_posterior(strain=data)
    bilby_posterior = sampler.to_bilby()
    sampler.plot_corner()
    


if __name__ == '__main__':
    pytest.main([__file__])
