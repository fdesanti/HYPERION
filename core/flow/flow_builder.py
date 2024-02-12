import os
import json
import numpy as np
import torch
import torch.nn as nn


from . import Flow
from . import CouplingTransform, AffineCouplingLayer, RandomPermutation
from ..distributions   import MultivariateNormalBase
from ..neural_networks import EmbeddingNetwork


from ...config import CONF_DIR

def build_flow( model_hyperparams           :dict = None,
                flow_kwargs                 :dict = None,
                coupling_layers_kwargs      :dict = None,
                prior_kwargs                :dict = None,
                flow_network_kwargs         :dict = None,
                embedding_network_kwargs    :dict = None,
                checkpoint_path                   = None,
               ):

    #loading a saved model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model_hyperparams = checkpoint['model_hyperparams']
        kwargs = checkpoint['configuration']

    #building model from scratch
    else:
        assert model_hyperparams is not None, 'Unable to build Flow since no hyperparams are passed'
    
        # First, read the JSON
        default_config_file = CONF_DIR + '/flow_config.json'
        with open(default_config_file) as json_file:
            kwargs = json.load(json_file)

        if flow_kwargs is not None:
            kwargs['flow'].update(flow_kwargs)
        if coupling_layers_kwargs is not None:
            kwargs['coupling_layers'].update(coupling_layers_kwargs)
        if prior_kwargs is not None:
            kwargs['prior'].update(prior_kwargs)
        if flow_network_kwargs is not None:
            kwargs['flow_network'].update(flow_network_kwargs)
        if embedding_network_kwargs is not None:
            kwargs['embedding_network'].update(embedding_network_kwargs)
    
    flow_kwargs              =  kwargs['flow']
    coupling_layers_kwargs   =  kwargs['coupling_layers']
    prior_kwargs             =  kwargs['prior']
    flow_network_kwargs      =  kwargs['flow_network']
    embedding_network_kwargs =  kwargs['embedding_network'] 
    

    #NEURAL NETWORK ---------------------------------------    
    embedding_network   = EmbeddingNetwork(**embedding_network_kwargs).float()
       
    #BASE DIST ----------------------------------------------------------------------------
    base = MultivariateNormalBase(**prior_kwargs)
    

    #COUPLING TRANSFORM ----------------------------------------------------------------
    coupling_layers = []
    for i in range(flow_kwargs['num_coupling_layers']):
        
        coupling_layers += [RandomPermutation(num_features=coupling_layers_kwargs['num_features'])]

        coupling_layers += [AffineCouplingLayer(coupling_layers_kwargs['num_features'], flow_network_kwargs['strain_features'], coupling_layers_kwargs['num_identity'], coupling_layers_kwargs['num_transformed'])]
        
    coupling_transform = CouplingTransform(coupling_layers)

    #FLOW --------------------------------------------------------------------------------------
    flow = Flow(base_distribution = base, 
                transformation    = coupling_transform, 
                embedding_network = embedding_network, 
                model_hyperparams = model_hyperparams).float()
    
    """loading (eventual) weights"""
    if checkpoint_path is not None:
        flow.load_state_dict(checkpoint['model_state_dict'])
        print('----> Model weights loaded!\n')
        model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'----> Flow has {params/1e6:.1f} M trained parameters')
        
    return flow

