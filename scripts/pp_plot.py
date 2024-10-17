import os
import sys
import yaml
import torch
import bilby

from tqdm import tqdm
from optparse import OptionParser
from tensordict import TensorDict

from hyperion.training import DatasetGenerator
from hyperion.simulations import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)
from hyperion import PosteriorSampler
from bilby.core.result import make_pp_plot


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--num_posteriors", default=128, help="Number of posteriors to sample for the pp plot")
    parser.add_option("-s", "--num_posterior_samples", default=50_000, help="Number of posterior samples to draw")
    parser.add_option("-v", "--verbosity", default=False, action="store_true", help="Verbosity of the flow sampler. (Default: False)")
    parser.add_option("-m", "--model_name", default='BHBH', help="Name of the model to sample from")
    
    (options, args) = parser.parse_args()
    
    NUM_POSTERIORS = int(options.num_posteriors)
    NUM_SAMPLES    = int(options.num_posterior_samples)
    VERBOSITY      = options.verbosity
    MODEL_NAME      = options.model_name
    
    #Setup & load model --------------------------------------------------
    conf_yaml = f'training_results/{MODEL_NAME}/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)

    WAVEFORM_MODEL = conf['waveform_model']
    PRIOR_PATH = f'training_results/{MODEL_NAME}/prior.yml'
    DURATION  = conf['duration']


    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
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
                                            fmin          = conf['fmin'],
                                            duration      = DURATION,
                                            reference_run = conf['ASD_reference_run'])
        
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs']
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, fs=conf['fs'], duration=DURATION, **wvf_kwargs)

        #setup dataset generator
        dataset_kwargs = {'waveform_generator'    : waveform_generator, 
                            'asd_generators'      : asd_samplers,
                            'det_network'         : det_network,
                            'num_preload'         : 2*NUM_POSTERIORS,
                            'device'              : device,
                            'batch_size'          : 1,
                            'inference_parameters': conf['inference_parameters'],
                            'prior_filepath'      : PRIOR_PATH,
                            'n_proc'              : 1,
                            'use_reference_asd'   : conf['use_reference_asd'],
                            'whiten_kwargs'       : conf['training_options']['whiten_kwargs']}

        test_ds = DatasetGenerator(**dataset_kwargs)
        test_ds.preload_waveforms()
        
        #set up Sampler
        checkpoint_path = f'training_results/{MODEL_NAME}/{MODEL_NAME}_flow_model.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path  = checkpoint_path, 
                                   waveform_generator    = waveform_generator,
                                   num_posterior_samples = NUM_SAMPLES,
                                   device                = device)
         

        #SAMPLING --------
        
        bilby_posterior_list = []
        
        #setup a fake bilby prior ---> used only to get the latex names in the PP Plot
        #(otherwise bilby gets upset)
        
        fake_bilby_priors = {}
        for key in conf['inference_parameters']:
            fake_bilby_priors[key] = bilby.core.prior.Uniform(0, 1, latex_label=sampler.latex_labels()[key])

        for _ in tqdm(range(NUM_POSTERIORS)):
        
            #simulate a signal
            parameters, whitened_strain, asd = test_ds.__getitem__(add_noise=conf['training_options']['add_noise'])

            #sample posterior
            posterior = sampler.sample_posterior(strain             = whitened_strain,
                                                 verbose            = VERBOSITY,
                                                 restrict_to_bounds = True)
        
            #get true parameters
            true_parameters = sampler.flow._post_process_samples(parameters, restrict_to_bounds=False)
            true_parameters = TensorDict.from_dict(true_parameters)
            true_parameters = {key: true_parameters[key].cpu().item() for key in true_parameters.keys()}

            #convert to bilby posterior
            bilby_posterior = sampler.to_bilby(priors=fake_bilby_priors, injection_parameters=true_parameters)
            
            bilby_posterior_list.append(bilby_posterior)
            
        #make pp plot
        pp_plot, pvals = make_pp_plot(bilby_posterior_list, filename=f'training_results/{MODEL_NAME}/pp_plot.png')
        

        
        
        

