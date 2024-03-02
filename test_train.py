import json
import torch
import corner
import numpy as np
import matplotlib.pyplot as plt

from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow
from hyperion.inference import ImportanceSampling

from gwskysim.gwskysim.sources import EffectiveFlyByTemplate
from gwskysim.gwskysim.detectors import GWDetector


if __name__ == '__main__':

    conf_json = CONF_DIR + '/train_config.json'
    
    with open(conf_json) as json_file:
        conf = json.load(json_file)

    
    
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
        plt.savefig('training_results/strain.png', dpi=200)

        #set up Flow model
        checkpoint_path = 'training_results/BHBH/BHBH_flow_model.pt'
        flow = build_flow(checkpoint_path=checkpoint_path).to(device)

 
        with torch.inference_mode():
            flow.eval()
            #initialize cuda (for a faster sampling)
            embedded_strain = flow.embedding_network(strain) 
            #print()
    
            posterior_samples = flow.sample(num_samples, strain,
                                        restrict_to_bounds=True,
                                       )
        
        parameters = flow._post_process_samples(parameters, restrict_to_bounds=False)
        #print(posterior_samples)
        
        posterior_dict={}
        truths = dict()

        plt.figure(figsize=(8, 40))
        print('\nparameter', 'true', 'flow median')
        for i, parameter in enumerate(posterior_samples.keys()):
            plot_samples = posterior_samples[parameter].cpu().numpy()
            posterior_dict[parameter] = plot_samples
            truths[parameter] = parameters[parameter].cpu().item()

            plt.subplot(10, 1, i+1)
            plt.hist(plot_samples, 'auto');
            plt.axvline(parameters[parameter].cpu().numpy(), color='r')
            plt.title(parameter)
            print(f'{parameter} - real: {parameters[parameter].item():.3f} - median flow: {np.median(plot_samples):.3f}')
        
            
        plt.savefig('training_results/flow_result.png')
        plt.close()
        
        plt.figure()
        corner.corner(posterior_dict, truths = truths, truth_color='tab:orange', color='#0072C1', 
                       quantiles=[0.16, 0.5, 0.84],
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                        plot_density=False, plot_datapoints=True, fill_contours=True,
                        show_titles=True, bins = 50, smooth=0.9
                        )

        plt.savefig('training_results/corner.png', dpi = 200)
        plt.close()

        print('\n\nflow prior metadata')
        print(flow.prior_metadata)

        IS = ImportanceSampling(flow=flow, waveform_generator=efbt, device=device)

        IS.compute_Bayes_factor(strain=strain)
        
