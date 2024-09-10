import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

from optparse import OptionParser

from tensordict import TensorDict
from hyperion.training import *
from hyperion.config import CONF_DIR
from hyperion.core.flow import build_flow
from hyperion.simulations import (ASD_Sampler, 
                                  GWDetectorNetwork, 
                                  WaveformGenerator)
from gwpy.timeseries import TimeSeries
from gwosc.datasets import event_gps

#from pycbc.types import TimeSeries as pycbcTimeSeries
from pycbc.psd import welch, interpolate


from hyperion.core import PosteriorSampler
from hyperion.core.fft import rfft, irfft

from tqdm import tqdm 
import numpy as np

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--num_posterior_samples", default=50_000, help="Number of posterior samples to draw")
    parser.add_option("-m", "--MODEL_DIR", default='training_results/BHBH', help="Directory containing the model to sample from")
    parser.add_option("-t", "--dt", default=0.25, help="Offset to apply to the GPS time of the event")

    (options, args) = parser.parse_args()
    
    NUM_SAMPLES    = int(options.num_posterior_samples)
    MODEL_DIR      = options.MODEL_DIR
    dt_gps         = float(options.dt)

    print(f'----> Running model saved at {MODEL_DIR}')

    conf_yaml = MODEL_DIR + '/hyperion_config.yml'
    
    with open(conf_yaml, 'r') as yaml_file:
        conf = yaml.safe_load(yaml_file)

    WAVEFORM_MODEL = conf['waveform_model']
    PRIOR_PATH = os.path.join(MODEL_DIR, 'prior.yml')
    DURATION  = conf['duration']

    detectors = conf['detectors']

    
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device = f'cuda:{num_gpus-1}'
        #device = 'cuda'
    else:
        device = 'cpu'
    
    event_gps = 1242442967.4 
    gps = 1242442967.4 #event_gps("GW190521")
    print('GW190521 GPS time', gps)
    start = int(gps) - 32
    end   = int(gps) + 8
    
    sampling_frequency = conf['fs']
    
    gps-=float(dt_gps)#0.12
    
    t0=gps-DURATION/2
    t1=gps+DURATION/2

    strain = dict()
    for det in tqdm(detectors):
        fname = f'gw190521_data/GW190521_data_{det}_start_{start}_end_{end}.csv'
        #
        try:
            strain[det] = TimeSeries.read(fname)#.resample(2048)
        except:
            strain[det] = TimeSeries.fetch_open_data(det, start, end, cache=True)
            strain[det].write(fname)
            
    

    whitened_strain = dict()
    asds = dict()
    torch_asd = dict()
    torch_psd = dict()
    torch_noisy_strain = dict()
    torch_whitened_strain = dict()
    
    for det in tqdm(detectors):
         asds[det] = strain[det].crop(start, end).asd(4,2, detrend='linear').to_pycbc()
         #torch_asd[det] = ASD_Sampler(det, reference_run = 'O3a', fs = 2048, duration=1).asd_reference
         
         #event_asd = strain[det].asd(4,2)
         #whitened_strain[det] = strain[det].whiten(8, detrend='linear').resample(2048).crop(t0, t1).copy()
         s = strain[det].resample(sampling_frequency).to_pycbc()
         torch_noisy_strain[det] = torch.from_numpy(s.time_slice(t0, t1).numpy()).to(device)
         psd = interpolate(welch(s, seg_len=2048), 1.0 / s.duration)
         whitened_strain[det] = (s.to_frequencyseries() / psd ** 0.5).to_timeseries().time_slice(t0, t1) * np.sqrt(2/sampling_frequency) 
         
         #interpolate psd
         f = np.fft.fftfreq(len(whitened_strain[det]), 1/sampling_frequency)
         interp_psd = np.interp(f, psd.sample_frequencies, psd)
         torch_psd[det] = torch.from_numpy(interp_psd).to(device)
         
         
         #whitened_strain[det] = 2*strain[det].whiten(asd = asds[det]).resample(2048).crop(t0, t1).copy() #/ np.sqrt(2/2048) 
         whitened_strain[det] -= whitened_strain[det].numpy().mean()
         torch_whitened_strain[det] = torch.from_numpy(whitened_strain[det].numpy()).to(device).float()
         #whitened_strain[det] = whitened_strain[det].copy()
         #whitened_strain[det].plot(epoch = gps)
         plt.figure(figsize=(20, 5))
         plt.plot(whitened_strain[det])
         plt.savefig(f'{MODEL_DIR}/{det}_whitened_strain.png')
         plt.close()
    torch_whitened_stacked_strain = torch.stack([torch_whitened_strain[det] for det in detectors]).unsqueeze(0).to(device).float()
    print(whitened_strain)
    #torch_asd = torch.stack([torch_asd[det] for det in torch_asd.keys()], dim=0).unsqueeze(0).to(device).float()
    #print(torch_asd.shape)
   

    with torch.device(device):

        #set up gwskysim detectors and asd_samplers
        det_network = GWDetectorNetwork(conf['detectors'], use_torch=True, device=device)
        det_network.set_reference_time(gps)
        
        asd_samplers = dict()
        for ifo in conf['detectors']:
            asd_samplers[ifo] = ASD_Sampler(ifo, device=device, fs=conf['fs'],fmin = conf['fmin'], duration=DURATION)
        
        with open(PRIOR_PATH, 'r') as f:
            prior_conf = yaml.safe_load(f)
            wvf_kwargs = prior_conf['waveform_kwargs']
        
        waveform_generator = WaveformGenerator(WAVEFORM_MODEL, 
                                               fs=conf['fs'], 
                                               duration=DURATION,
                                               det_network=det_network,
                                               **wvf_kwargs)
        
        #SAMPLING --------
        #parameters, strain = test_ds.__getitem__()
        '''
        plt.figure(figsize=(20, 15))
        for i, det in enumerate(det_network.detectors):
            plt.subplot(3, 1, i+1)
            plt.plot(strain[0][i].cpu().numpy())
            plt.title(det)            
        plt.savefig('training_results/BHBH/strain.png', dpi=200)
        '''

        #set up Sampler
        checkpoint_path = f'{MODEL_DIR}/BHBH_flow_model.pt'
        
        sampler = PosteriorSampler(flow_checkpoint_path  = checkpoint_path, 
                                   waveform_generator    = waveform_generator,
                                   num_posterior_samples = NUM_SAMPLES,
                                   device                = device)

        #print(sampler.flow.configuration)     
        #print(sampler.flow.base_distribution)   

        posterior = sampler.sample_posterior(strain = torch_whitened_stacked_strain,#/np.sqrt(2/2048),
                                             #asd               = torch_asd,
                                             num_samples        = NUM_SAMPLES,
                                             restrict_to_bounds = True,
                                             event_time         = event_gps)
        
        
        '''
        from astropy.cosmology import Planck18, z_at_value
        import astropy.units as u
        if 'luminosity_distance' in posterior.keys():
            z = z_at_value(Planck18.luminosity_distance, posterior['luminosity_distance'].cpu()*u.Mpc)
        elif 'distance' in posterior.keys():
            z = z_at_value(Planck18.luminosity_distance, posterior['distance'].cpu()*u.Mpc)
        
        if 'M' in posterior.keys():
            sampler.posterior['M_source'] = sampler.posterior['M']/torch.from_numpy((1+z)).to(device)
        
        if 'Mchirp' in sampler.posterior.keys():
            sampler.posterior['Mchirp_source'] = sampler.posterior['Mchirp']/torch.from_numpy((1+z)).to(device)
        else:
            sampler.posterior['Mchirp'] = sampler.posterior['M']**0.6/sampler.posterior['q']**0.2
        
        sampler.posterior['z'] = torch.from_numpy(z).to(device)
        '''
        #plot corner + skymap + save posterior samples
        sampler.plot_corner(figname=f'{MODEL_DIR}/gw190521_corner.png')
        sampler.to_bilby().save_posterior_samples(filename=f'{MODEL_DIR}/gw190521_posterior.csv')
        #sampler.plot_skymap(jobs=2, maxpts=NUM_SAMPLES)
        
        #plot reconstructed waveform
        posterior = sampler.posterior
        posterior['inclination']       = torch.zeros_like(posterior['inclination'])
        posterior['polarization']      = torch.zeros_like(posterior['polarization'])
        posterior['H_hyp']             = 1.0007*torch.ones_like(posterior['H_hyp'])
        posterior['coalescence_angle'] = torch.zeros_like(posterior['H_hyp'])
        
        
        asds = {det:asd_samplers[det].asd_reference.unsqueeze(0) for det in detectors}
        sampler.plot_reconstructed_waveform(whitened_strain=torch_whitened_strain, asd=asds, 
                                            CL=90)
              

        
        print('[INFO]: Peforming Importance Sampling...')
        posterior = sampler.sample_posterior(strain = torch_whitened_stacked_strain,#/np.sqrt(2/2048),
                                             verbose            = False,  
                                             #asd               = torch_asd,
                                             num_samples        = NUM_SAMPLES,
                                             restrict_to_bounds = False,
                                             event_time         = event_gps)
        
        is_kwargs = {'whitened_strain':torch_whitened_strain, 'strain':torch_noisy_strain, 'psd':torch_psd, 'event_time':gps}
        reweighted_poterior = sampler.reweight_posterior(importance_sampling_kwargs=is_kwargs, num_samples=NUM_SAMPLES)

        if 'luminosity_distance' in posterior.keys():
            z = z_at_value(Planck18.luminosity_distance, reweighted_poterior['luminosity_distance'].cpu()*u.Mpc)
        elif 'distance' in posterior.keys():
            z = z_at_value(Planck18.luminosity_distance, reweighted_poterior['distance'].cpu()*u.Mpc)
        
        if 'M' in posterior.keys():
            reweighted_poterior['M_source'] = reweighted_poterior['M']/torch.from_numpy((1+z)).to(device)
        
        if 'Mchirp' in sampler.posterior.keys():
            reweighted_poterior['Mchirp_source'] = reweighted_poterior['Mchirp']/torch.from_numpy((1+z)).to(device)
        else:
            reweighted_poterior['Mchirp'] = reweighted_poterior['M']**0.6/sampler.posterior['q']**0.2
        
        reweighted_poterior['z'] = torch.from_numpy(z).to(device)
        
        bilby_reweighted_posterior = sampler.to_bilby(posterior=reweighted_poterior).save_posterior_samples(filename=f'{MODEL_DIR}/gw190521_reweighted_posterior.csv')
        
        print(sampler.IS_results)

        valid_samples  = sampler.IS_results['stats']['valid_samples']
        log_prior      = sampler.IS_results['stats']['logP'][valid_samples]
        log_likelihood = sampler.IS_results['stats']['logL']
        log_posterior  = sampler.IS_results['stats']['log_posterior'][valid_samples]
        weights        = sampler.IS_results['stats']['weights']

        plt.figure()
        plt.scatter(log_posterior.cpu().numpy(), (log_prior+log_likelihood).cpu().numpy(),  c=weights.cpu().numpy(), cmap='viridis', s=1)
        plt.xlabel('log posterior')
        plt.ylabel('log prior + log likelihood')
        plt.colorbar()
        plt.savefig(f'{MODEL_DIR}/posterior_vs_prior_likelihood.png')
        plt.show()
        plt.close()

        sampler.plot_corner(posterior=reweighted_poterior, figname=f'{MODEL_DIR}/gw190521_corner_reweighted.png')
        sampler.plot_skymap(posterior=reweighted_poterior, jobs=2, maxpts=NUM_SAMPLES)
       

        '''
        #generate corner plot
        from astropy.cosmology import Planck18, z_at_value
        import astropy.units as u
        z = z_at_value(Planck18.luminosity_distance, posterior['distance'].cpu()*u.Mpc)
        
        #sampler.posterior['M'] = sampler.posterior['M']/torch.from_numpy((1+z)).to(device)
        sampler.plot_corner()
        bilby_posterior = sampler.to_bilby().save_posterior_samples(filename='training_results/BHBH/posterior.csv')
        sampler.plot_skymap(jobs=2, maxpts=1_000)
        '''

        