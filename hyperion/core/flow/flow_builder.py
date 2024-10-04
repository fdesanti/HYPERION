import yaml
import numpy as np
import torch

from . import Flow
from . import CouplingTransform, RandomPermutation, coupling_layer_dict
from ..utilities import GWLogger
from ..distributions import MultivariateNormalBase, base_distributions_dict
from ..neural_networks import EmbeddingNetwork, EmbeddingNetworkAttention, embedding_network_dict
from ...config import CONF_DIR

log = GWLogger()

def build_flow( prior_metadata           :dict = None,
                flow_kwargs              :dict = None,
                coupling_layers_kwargs   :dict = None,
                base_distribution_kwargs :dict = None,
                embedding_network_kwargs :dict = None,
                checkpoint_path                = None,
                config_file                    = None,
               ):

    #loading a saved model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        prior_metadata = checkpoint['prior_metadata']
        kwargs = checkpoint['configuration']

    #building model from scratch
    else:
        assert prior_metadata is not None, 'Unable to build Flow since no hyperparams are passed'
        # First, read the JSON
        config_file = CONF_DIR + '/hyperion_config.yml' if config_file is None else config_file
        with open(config_file, 'r') as yaml_file:
            kwargs = yaml.safe_load(yaml_file)

        if flow_kwargs is not None:
            kwargs['flow'].update(flow_kwargs)
        if coupling_layers_kwargs is not None:
            kwargs['coupling_layers'].update(coupling_layers_kwargs)
        if base_distribution_kwargs is not None:
            kwargs['base_distribution'].update(base_distribution_kwargs)
        if embedding_network_kwargs is not None:
            kwargs['embedding_network'].update(embedding_network_kwargs)
    
    flow_kwargs              =  kwargs['flow']
    coupling_layers_kwargs   =  kwargs['coupling_layers']
    base_distribution_kwargs =  kwargs['base_distribution']
    embedding_network_kwargs =  kwargs['embedding_network']['kwargs']
    embedding_network_model  =  kwargs['embedding_network']['model']

    configuration = kwargs
    
    #NEURAL NETWORK ---------------------------------------    
    #compute the shape of the strain tensor
    detectors = configuration['detectors']
    duration  = configuration['duration']
    fs        = configuration['fs']  
    strain_shape = [len(detectors), int(duration*fs)]
    embedding_network = embedding_network_dict[embedding_network_model](strain_shape=strain_shape, fs=fs, **embedding_network_kwargs).float()
       
    #BASE DIST ----------------------------------------------------------------------------
    #FIXME - add in deepfaset 
    try:
        dist_name = base_distribution_kwargs['dist_name']
        kw        = base_distribution_kwargs['kwargs']
        base      = base_distributions_dict[dist_name](**kw)
    except:
        base = MultivariateNormalBase(**base_distribution_kwargs)
    

    #COUPLING TRANSFORM ----------------------------------------------------------------
    coupling_kind = flow_kwargs.get('coupling', 'affine')
    CouplingLayer = coupling_layer_dict[coupling_kind]
    
    coupling_layers = []
    for _ in range(flow_kwargs['num_coupling_layers']):
        
        coupling_layers += [RandomPermutation(num_features=coupling_layers_kwargs['num_features'])]
        coupling_layers += [CouplingLayer(**coupling_layers_kwargs)]
        
    coupling_transform = CouplingTransform(coupling_layers)

    #FLOW --------------------------------------------------------------------------------------
    flow = Flow(base_distribution = base, 
                transformation    = coupling_transform, 
                embedding_network = embedding_network, 
                prior_metadata = prior_metadata, 
                configuration = configuration).float()
    
    """loading (eventual) weights"""
    if checkpoint_path is not None:
        flow.load_state_dict(checkpoint['model_state_dict'])
        print('\n')
        log.info('Model weights loaded!')
        model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        log.info(f'Flow has {params/1e6:.1f} M trained parameters')
        
    return flow

