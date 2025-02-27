"""Test script for flow module."""

import pytest
import torch

from torch.optim import Adam
from hyperion.core.flow import *
from hyperion.core.distributions import *


def _test_flow(coupling):
    """Test unconditional flow training."""
    prior = MultivariatePrior(dict(x=UniformPrior(-1, 1), y=UniformPrior(-1, 1)))

    #build the flow
    base_dist = MultivariateNormalBase(dim=2)
    coupling_layers = []
    for _ in range(4):
        coupling_layers.append(coupling_layer_dict[coupling](num_features=2))
        coupling_layers.append(RandomPermutation(num_features=2))

    flow = Flow(base_distribution = base_dist, 
                transformation    = CouplingTransform(coupling_layers),
                prior_metadata    = prior.metadata)
    
    #train the flow
    optimizer = Adam(flow.parameters(), lr=1e-3)

    for _ in range(5):
        optimizer.zero_grad()
        #sample the prior
        x = prior.sample(10, standardize=True).flatten().view(-1, 2)
        loss = -flow.log_prob(x).mean()
        loss.backward()
        optimizer.step()

    #test the flow
    flow.eval()
    with torch.inference_mode():
        #test the flow log_prob
        x = prior.sample(10, standardize=True).flatten().view(-1, 2)
        log_prob = flow.log_prob(x)
        assert log_prob.shape == (10,)
        assert log_prob.dtype == torch.float32

        #test the flow sampling
        samples = flow.sample(10).flatten()
        assert samples.shape == (10, 2)

def test_unconditional_flow():
    couplings = list(coupling_layer_dict.keys())
    for coupling in couplings:
        _test_flow(coupling)

if __name__ == '__main__':
    pytest.main([__file__])
