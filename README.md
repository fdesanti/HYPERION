![Test Status](https://img.shields.io/badge/Tests-Passed-brightgreen)

# `HYPERION` - HYPer-fast close EncounteR Inference from Observations with Normalizing-flows 

HYPERION is pipeline for the detection and Parameter Estimation of gravitational waves from Close Encounters (see e.g. https://arxiv.org/abs/1909.02143) based on Normalizing Flows (https://arxiv.org/abs/1912.02762) 

## Installation

To install hyperion you can clone this repository then install it

```
git clone https://github.com/fdesanti/HYPERION.git
cd HYPERION
python setup.py install
```

### PyTorch installation
By default (on Linux) PyTorch is shipped with the latest CUDA binaries. 
Refer to the page https://pytorch.org/get-started/locally/ for other CUDA versions and/or installation
options.

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

It is therefore recommended to install pytorch before installing the requirements


### TEOBResumS-DALI installation
TEOBResumS-DALI can be installed with the following steps.

```
git clone https://bitbucket.org/teobresums/teobresums.git
cd teobresums
git checkout DALI
cd Python
python setup.py build_ext --inplace
```
Then export the path to the Python folder to your PYTHONPATH. 

```
echo PYTHONPATH="<path_to_Python_folder>:$PYTHONPATH" > ~/.bashrc
echo export PYTHONPATH > ~/.bashrc
source ~/.bashrc
```

Substitute ~/.bashrc with ~/.zprofile if you're on MacOS

## Usage

Once the steps above have been completed hyperion can be imported and directly used. 
As an example: a typical inference code would be:

```python
from hyperion.core import PosteriorSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#initialize the Posterior Sampler
sampler = PosteriorSampler(flow_checkpoint_path=path_to_trained_model, 
                           waveform_generator=efbt, 
                           num_posterior_samples=num_samples, 
                           device=device)

#sample posterior
posterior = sampler.sample_posterior(strain=whitened_strain,
                                     restrict_to_bounds=True)

#Once the posterior is sampled it is also possible to make corner plots
sampler.plot_corner(injection_parameters=true_parameters)


#The sampler class allows also to perform the Importance Sampling and computing Bayes Factors
is_kwargs = {'whitened_strain':whitened_strain, 'strain':noisy_strain, 'psd':psd, 'event_time':t_gps}
resampled_posterior = flow_sampler.reweight_posterior(posterior=posterior,                      
                                                      importance_sampling_kwargs=is_kwargs)

print(f'The signal vs noise Bayes Factor is {sampler.BayesFactor:.2f}')
```

## Citation

If you use `HYPERION`, please cite the 
[corresponding paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.102004) as follows.

> De Santi et al., Physical Review D **109**, 102004 (2024)

**Bibtex**

```
@article{DeSanti2024,
  title = {Deep learning to detect gravitational waves from binary close encounters: Fast parameter estimation using normalizing flows},
  author = {De Santi, Federico and Razzano, Massimiliano and Fidecaro, Francesco and Muccillo, Luca and Papalini, Lucia and Patricelli, Barbara},
  journal = {Phys. Rev. D},
  volume = {109},
  issue = {10},
  pages = {102004},
  numpages = {21},
  year = {2024},
  month = {May},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevD.109.102004},
  url = {https://link.aps.org/doi/10.1103/PhysRevD.109.102004}
} 