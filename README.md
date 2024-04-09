# `HYPERION` - HYPer-fast close EncounteR Inference from Observations with Normalizing-flows 

HYPERION is pipeline for the detection and Parameter Estimation of gravitational waves from Close Encounters (see e.g. https://arxiv.org/abs/1909.02143) based on Normalizing Flows (https://arxiv.org/abs/1912.02762) 

## Installation

Clone the repo in a given folder (e.g. the "work" folder)

```
git clone git@gitlab.com:mmagwpisa/hyperion.git
```

then install the dependencies.

```
pip install -r requirements.txt
```

To use HYPERION as an installed package you need to add it to PYTHONPATH. 
Run the following (on Mac change "bashrc" with "zprofile")

```
echo PYTHONPATH = "path_to_work_dir":$PYTHONPATH > ~/.bashrc  
echo export PYTHONPATH > ~/.bashrc  
source ~/.bashrc
```

### A note about PyTorch
By default PyTorch will download (on Linux) the latest CUDA binaries (12.1). 
Refer to the page https://pytorch.org/get-started/locally/ for other CUDA versions
(e.g. PcUniverse only works with CUDA <= 11.8)

## Usage

Once the steps above have been completed hyperion can be imported and directly used. 
As an example: a typical inference code would be:

```python
from hyperion.core import PosteriorSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#initialize the Posterior Sampler
sampler = PosteriorSampler(flow_checkpoint_path=path_to_trained_model, 
                           waveform_generator = efbt, 
                           num_posterior_samples=num_samples, 
                           device=device)

#sample posterior
posterior = sampler.sample_posterior(strain = whitened_strain,
                                     restrict_to_bounds=True)

#Once the posterior is sampled it is also possible to make corner plots
sampler.plot_corner(injection_parameters=true_parameters)


#The sampler class allows also to perform the Importance Sampling and computing Bayes Factors
is_kwargs = {'whitened_strain':whitened_strain, 'strain':noisy_strain, 'psd':psd, 'event_time':t_gps}
resampled_posterior = flow_sampler.reweight_posterior(posterior=posterior,                      
                                                      importance_sampling_kwargs=is_kwargs)

print(f'The signal vs noise Bayes Factor is {sampler.BayesFactor:.2f}')
```




