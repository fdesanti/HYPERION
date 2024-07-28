import os
import yaml
import torch
import matplotlib.pyplot as plt

from tensordict import TensorDict
from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow
from hyperion.simulations import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)

from hyperion.core import PosteriorSampler


if __name__ == '__main__':

    conf_yaml = CONF_DIR + '/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)

    WAVEFORM_MODEL = conf['waveform_model']
    PRIOR_PATH = os.path.join(CONF_DIR, conf['prior']+'.yml')
    DURATION  = conf['duration']

    
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
        #device = 'cuda'
    else:
        device = 'cpu'
    
   

    with torch.device(device):

        #set up gwskysim detectors and asd_samplers
        det_network = GWDetectorNetwork(conf['detectors'], use_torch=True, device=device)
        det_network.set_reference_time(conf['reference_gps_time'])
        
        asd_samplers = dict()
        for ifo in conf['detectors']:
            asd_samplers[ifo] = ASD_Sampler(ifo, 
                                            device=device, 
                                            fs=conf['fs'], 
                                            duration=DURATION, 
                                            reference_run=conf['ASD_reference_run'])
        
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs']
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, fs=conf['fs'], duration=DURATION, **wvf_kwargs)

        #setup dataset generator
        dataset_kwargs = {'waveform_generator': waveform_generator, 
                            'asd_generators':asd_samplers, 
                            'det_network': det_network,
                            'num_preload': 2,
                            'device':device, 
                            'batch_size': 1, 
                            'inference_parameters': conf['inference_parameters'],
                            'prior_filepath': PRIOR_PATH, 
                            'n_proc': 1, 
                            'use_reference_asd': conf['use_reference_asd']}

        test_ds = DatasetGenerator(**dataset_kwargs)
        test_ds.preload_waveforms()
         

        #SAMPLING --------
        num_samples = 50_000
        parameters, strain, asd = test_ds.__getitem__()
        #print(asd.shape)
        plt.figure(figsize=(20, 15))
        for i, det in enumerate(det_network.detectors):
            plt.subplot(3, 1, i+1)
            plt.plot(strain[0][i].cpu().numpy())
            plt.title(det)           
        plt.show()
        plt.savefig('training_results/BHBH/strain.png', dpi=200)

        #set up Sampler
        checkpoint_path = 'training_results/BHBH/BHBH_flow_model.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path=checkpoint_path, 
                                   waveform_generator=waveform_generator, 
                                   num_posterior_samples=num_samples, 
                                   device=device)
        sampler.flow.eval()

        posterior = sampler.sample_posterior(strain = strain, asd = asd, num_samples=num_samples, restrict_to_bounds = True)
        
        #compare sampled parameters to true parameters
        true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
        true_parameters = TensorDict.from_dict(true_parameters)
        true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

        print('Sampled parameters vs true parameters medians')
        for par in true_parameters:
            print(f"{par}: {posterior[par].cpu().median():.2f} vs {true_parameters[par]:.2f}")
        
        #generate corner plot
        sampler.plot_corner(injection_parameters=true_parameters)
