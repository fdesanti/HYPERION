import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow

from gwskysim.gwskysim.sources import EffectiveFlyByTemplate
from gwskysim.gwskysim.detectors import GWDetector


if __name__ == '__main__':

    conf_json = CONF_DIR + '/train_config.json'
    
    with open(conf_json) as json_file:
        conf = json.load(json_file)

    
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
   

    with torch.device(device):

        #set up gwskysim detectors and asd_samplers
        detectors = dict()
        asd_samplers = dict()
        for ifo in conf['detectors']:
            detectors[ifo] = GWDetector(ifo, use_torch=True, device = device)
            asd_samplers[ifo] = ASD_sampler(ifo, device=device, fs=conf['fs'])
        
        #set up EFB-T waveform
        efbt = EffectiveFlyByTemplate(duration = 1, torch_compile=False, detectors=detectors, fs=conf['fs'], device = device)

        #setup dataset generator
        dataset_kwargs = {'waveform_generator': efbt, 'asd_generators':asd_samplers, 
                          'device':device, 'batch_size': 1}

        test_ds = DatasetGenerator(**dataset_kwargs, random_seed=conf['test_seed'])
          
        
        #set up Flow model
        checkpoint_path = 'training_results/BHBH/BHBH_flow_model.pt'
        flow = build_flow(checkpoint_path=checkpoint_path).to(device)

        #SAMPLING --------
        num_samples = 50_000
        parameters, strain = test_ds.__getitem__()
        with torch.no_grad():
            flow.eval()
            #initialize cuda (for a faster sampling)
            flow.embedding_network(torch.empty(strain.shape)) 
    
            posterior_samples = flow.sample(num_samples, strain,
                                        restrict_to_bounds=False,
                                       )

        parameters = flow._post_process_samples(parameters, restrict_to_bounds=False)
        
        print(posterior_samples)

        plt.figure(figsize=(8, 40))
        print('parameter', 'true', 'flow median')
        for i, parameter in enumerate(posterior_samples.keys()):
            plot_samples = posterior_samples[parameter].cpu().numpy()

            plt.subplot(10, 1, i+1)
            plt.hist(plot_samples, 'auto');
            plt.axvline(parameters[parameter].cpu().numpy(), color='r')
            plt.title(parameter)
            print(parameter, parameters[parameter], np.median(plot_samples))
        
            
        plt.savefig('training_results/flow_result.png')
        plt.close()

        
    
    
