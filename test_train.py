import yaml
import torch
import matplotlib.pyplot as plt

from tensordict import TensorDict

from hyperion.config import CONF_DIR
from hyperion.training import *
from hyperion.training.dataset.dataset_generator import DatasetGenerator

from hyperion.simulations import GWDetector, ASD_Sampler
from hyperion.simulations.waveforms import EffectiveFlyByTemplate

from hyperion.core import PosteriorSampler


if __name__ == '__main__':

    conf_yaml = CONF_DIR + '/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)

    
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
    else:
        device = 'cpu'
    
   

    with torch.device(device):

        #set up gwskysim detectors and asd_samplers
        detectors = dict()
        asd_samplers = dict()
        for ifo in conf['detectors']:
            detectors[ifo] = GWDetector(ifo, use_torch=True, device = device)
            asd_samplers[ifo] = ASD_Sampler(ifo, device=device, fs=conf['fs'])
        
        #set up EFB-T waveform
        efbt = EffectiveFlyByTemplate(duration = 1, torch_compile=False, detectors=detectors, fs=conf['fs'], device = device)

        #setup dataset generator
        dataset_kwargs = {'waveform_generator': efbt, 'asd_generators':asd_samplers, 
                          'device':device, 'batch_size': 1, 
                          'inference_parameters': conf['inference_parameters']}

        test_ds = DatasetGenerator(**dataset_kwargs)
         

        #SAMPLING --------
        num_samples = 50_000
        parameters, strain = test_ds.__getitem__()
        plt.figure(figsize=(20, 15))
        for i, det in enumerate(detectors.keys()):
            plt.subplot(3, 1, i+1)
            plt.plot(strain[0][i].cpu().numpy())
            plt.title(det)            
        plt.savefig('training_results/BHBH/strain.png', dpi=200)

        #set up Sampler
        checkpoint_path = 'training_results/BHBH/BHBH_flow_model.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path=checkpoint_path, waveform_generator=efbt, num_posterior_samples=num_samples, device=device)

        posterior = sampler.sample_posterior(strain = strain, num_samples=num_samples, restrict_to_bounds = True)
        
        #compare sampled parameters to true parameters
        true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
        true_parameters = TensorDict.from_dict(true_parameters)
        true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

        print('Sampled parameters vs true parameters medians')
        for par in true_parameters:
            print(f"{par}: {posterior[par].cpu().median():.2f} vs {true_parameters[par]:.2f}")
        
        #generate corner plot
        sampler.plot_corner(injection_parameters=true_parameters)