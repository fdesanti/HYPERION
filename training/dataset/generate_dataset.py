import torch
import h5py
import json
from tqdm import tqdm
from optparse import OptionParser
from hyperion.config import CONF_DIR
from hyperion.core.distributions import *
from gwskysim.gwskysim.sources import EffectiveFlyByTemplate

from tensordict import TensorDict

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-o", "--output", default=None, help="Output Directory")
    parser.add_option("-b", "--batch_size", default=10_000, help="Batch size for parallel efbt computations (on GPU/CPU). (Default: 10_000)")
    parser.add_option("-d", "--device", default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help="Device to be used to generate the waveforms. (Default: cuda if GPU is available, otherwise CPU)")
    parser.add_option("-s", "--seed", default=123, help="Random seed. (Default: 123)")
    parser.add_option("-m", "--mode", default='training', help='The dataset to simulate: either <training>, <validation> or <test>')
    parser.add_option("-n", "--N", default=1e6, help='Number of samples to simulate')

    (options, args) = parser.parse_args()
    
    output_dir = options.output
    device     = options.device
    batch_size = int(options.batch_size)
    N = int(options.N)
    mode = options.mode
    seed = int(options.seed)
    if mode == 'validation':
        seed+=10
    elif mode == 'test':
        seed+=100
        

    
    
    
    conf_json = CONF_DIR + '/BHBH-CE_population.json'
    
    with open(conf_json) as json_file:
        prior_kwargs = json.load(json_file)
        
    
    efbt = EffectiveFlyByTemplate(duration = 1, torch_compile=False, fs=2048, device = device)
    
    prior = dict()
    for i, p in enumerate(prior_kwargs['parameters'].keys()):
        dist = prior_kwargs['parameters'][p]['distribution']
        if dist == 'delta':
            val = prior_kwargs['parameters'][p]['value']
            prior[p] = prior_dict_[dist](val, device, seed+i)
        else:
            min, max = prior_kwargs['parameters'][p]['min'], prior_kwargs['parameters'][p]['max']
            prior[p] = prior_dict_[dist](min, max, device, seed+i)
    
    
    #convert prior dictionary to MultivariatePrior
    multi_prior = MultivariatePrior(prior)
    all_samples = TensorDict.from_dict(multi_prior.sample((N, 1)))

    
    num_batches = N // batch_size
    remaining = N - num_batches*batch_size
    
    #open the output file
    with h5py.File(f'{mode}_dataset.hdf5', 'w') as hf:
        
        #save the parameters
        save_p = dict(all_samples)
        
        parameters = hf.create_group('parameters')
        for p in ['m1', 'm2', 'e0', 'p_0', 'polarization', 'inclination', 'distance']:
            parameters[p] = save_p[p].squeeze(1).cpu().numpy()
        
    
        for ibatch in tqdm(range(num_batches)):
        
            #slice samples
            samples = all_samples[ibatch*batch_size:(ibatch+1)*batch_size]
            
            #compute waveforms on device
            h = efbt(**samples)
            
            #transfer both waveforms and prior samples on cpu
            for key in h.keys():
                h[key] = h[key].cpu().numpy()
            
            for key in samples.keys():
                samples[key] = samples[key].cpu()
                
            save_p = {}
            for i in range(batch_size):
                
                waveforms = hf.create_group(str(i+ibatch*batch_size))
                
                waveforms.create_dataset('hp', data = h['hp'][i], dtype = 'float32')
                waveforms.create_dataset('hc', data = h['hc'][i], dtype = 'float32')
                    
            
                
    with h5py.File(f'{mode}_dataset.hdf5', 'r') as hf:
        #print(hf.keys())   
        print(hf['0'].keys())  
        print(hf['parameters'].keys()) 
        print(hf['parameters']['m1'])          
        
    
        
        
    
            
    
    
    
       
        
        
    
    
    
