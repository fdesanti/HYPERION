import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

from optparse import OptionParser
from tensordict import TensorDict
from hyperion.training import DatasetGenerator
from hyperion.simulations import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)

from hyperion.core import PosteriorSampler


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--num_posterior_samples", default=50_000, help="Number of posterior samples to draw")
    parser.add_option("-v", "--verbosity", default=False, action="store_true", help="Verbosity of the flow sampler. (Default: False)")
    parser.add_option("-m", "--model_dir", default='training_results/BHBH', help="Directory containing the model to sample from")
    
    (options, args) = parser.parse_args()
    
    NUM_SAMPLES    = int(options.num_posterior_samples)
    VERBOSITY      = options.verbosity
    MODEL_DIR      = options.model_dir
    
    #Setup & load model --------------------------------------------------
    conf_yaml = MODEL_DIR + '/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)


    WAVEFORM_MODEL = conf['waveform_model']
    PRIOR_PATH = os.path.join(MODEL_DIR, 'prior.yml')
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
                                            device        = device,
                                            fs            = conf['fs'],
                                            duration      = DURATION,
                                            reference_run = conf['ASD_reference_run'])
        
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs']
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, 
                                               fs=conf['fs'], 
                                               duration=DURATION, 
                                               det_network=det_network,
                                               **wvf_kwargs)

        #Setup dataset generator
        dataset_kwargs = {  'waveform_generator'  : waveform_generator, 
                            'asd_generators'      : asd_samplers,
                            'det_network'         : det_network,
                            'num_preload'         : 2,
                            'device'              : device,
                            'batch_size'          : 1,
                            'inference_parameters': conf['inference_parameters'],
                            'prior_filepath'      : PRIOR_PATH,
                            'n_proc'              : 1,
                            'use_reference_asd'   : conf['use_reference_asd'],
                            'whiten_kwargs'       : conf['training_options']['whiten_kwargs']}

        test_ds = DatasetGenerator(**dataset_kwargs)
        test_ds.preload_waveforms()
         

        #SAMPLING --------                
        parameters, whitened_strain, asd = test_ds.__getitem__(add_noise=conf['training_options']['add_noise'])

        
        #print(asd.shape)
        plt.figure(figsize=(20, 15))
        t = torch.arange(0, DURATION, 1/conf['fs']) - DURATION/2
        for i, det in enumerate(det_network.detectors):
            plt.subplot(3, 1, i+1)
            plt.plot(t.cpu().numpy(), whitened_strain[0][i].cpu().numpy())
            plt.title(det)           
        plt.savefig(f'{MODEL_DIR}/strain.png', dpi=200)
        plt.show()

        #set up Sampler
        checkpoint_path = f'{MODEL_DIR}/BHBH_flow_model.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path  = checkpoint_path, 
                                   waveform_generator    = waveform_generator,
                                   num_posterior_samples = NUM_SAMPLES,
                                   device                = device)

        posterior = sampler.sample_posterior(strain = whitened_strain, asd = asd, restrict_to_bounds = True)
        
        #compare sampled parameters to true parameters
        true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
        true_parameters = TensorDict.from_dict(true_parameters)
        true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

        print('Sampled parameters vs true parameters medians')
        for par in true_parameters:
            if par == 'H_hyp':
                print(f"{par}: {posterior[par].cpu().median():.5f} vs {true_parameters[par]:.5f}")
            else:
                print(f"{par}: {posterior[par].cpu().median():.2f} vs {true_parameters[par]:.2f}")
        
        #generate corner plot
        sampler.plot_corner(injection_parameters=true_parameters)
        
        #plot reconstructed_waveform
        if not 'j_hyp' in posterior.keys():
            posterior['j_hyp'] = torch.tensor([4.0]*len(posterior)).to(device)
        asd_dict={ifo: asd[0][i].unsqueeze(0) for i, ifo in zip(range(len(asd[0])), det_network.detectors)}
        whitened_strain = {ifo: whitened_strain[0][i] for i, ifo in zip(range(len(whitened_strain[0])), det_network.detectors)}
        sampler.plot_reconstructed_waveform(posterior=posterior, 
                                            whitened_strain=whitened_strain, 
                                            asd=asd_dict, 
                                            CL=50)
        