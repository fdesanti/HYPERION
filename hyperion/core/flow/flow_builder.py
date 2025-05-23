import yaml
import numpy as np
import torch

from . import GWFlow
from . import CouplingTransform, RandomPermutation, coupling_layer_dict
from ..utilities import HYPERION_Logger
from ..distributions import MultivariateNormalBase, base_distributions_dict
from ..neural_networks import EmbeddingNetwork, EmbeddingNetworkAttention, embedding_network_dict
from ...config import CONF_DIR

log = HYPERION_Logger()

def build_flow( metadata                 :dict = None,
                flow_kwargs              :dict = None,
                coupling_layers_kwargs   :dict = None,
                base_distribution_kwargs :dict = None,
                embedding_network_kwargs :dict = None,
                checkpoint_path                = None,
                config_file_path               = None,
               ):
    
    """
    Build a Flow object from scratch or from a saved model.

    Args:
        metadata           (dict): Metadata of the prior distribution
        flow_kwargs              (dict): Hyperparameters of the flow
        coupling_layers_kwargs   (dict): Hyperparameters of the coupling layers
        base_distribution_kwargs (dict): Hyperparameters of the base distribution
        embedding_network_kwargs (dict): Hyperparameters of the embedding network
        checkpoint_path           (str): Path to a saved model
        config_file_path               (str): Path to a configuration file
    
    Returns:
        Flow: a Flow object
    """
    
    #loading a saved model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        metadata = checkpoint['metadata']
        kwargs = metadata

    #building model from scratch
    else:
        assert metadata is not None, 'Unable to build Flow since no prior metadata was provided.'
        assert isinstance(metadata, dict), 'Please provide a dictionary with the prior metadata.'
        assert 'prior_metadata' in metadata, 'Please provide a dictionary with the prior_metadata key.'
        # First, read the JSON
        config_file_path = CONF_DIR + '/default_hyperion_config.yml' if config_file_path is None else config_file_path
        with open(config_file_path, 'r') as yaml_file:
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
    metadata.update(configuration)
    
    #NEURAL NETWORK ---------------------------------------    
    #compute the shape of the strain tensor
    detectors = configuration['detectors']
    duration  = configuration['duration']
    fs        = configuration['fs']  
    strain_shape = [len(detectors), int(duration*fs)]
    embedding_network = embedding_network_dict[embedding_network_model](strain_shape=strain_shape, fs=fs, **embedding_network_kwargs).float()
       
    #BASE DIST ----------------------------------------------------------------------------
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
    flow = GWFlow(base_distribution = base, 
                  transformation    = coupling_transform, 
                  embedding_network = embedding_network, 
                  metadata          = metadata
                 ).float()
    
    """loading (eventual) weights"""
    if checkpoint_path is not None:
        flow.load_state_dict(checkpoint['model_state_dict'])
        print('\n')
        log.info('Model weights loaded!')
        model_parameters = filter(lambda p: p.requires_grad, flow.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        log.info(f'Flow has {params/1e6:.1f} M trained parameters')
        
    return flow

