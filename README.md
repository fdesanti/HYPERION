![Test Status](https://img.shields.io/badge/Tests-Passed-brightgreen)

![Logo](docs/images/hyperion_logo.png)

# `HYPERION` - HYPER-fast Inference from Observations with Normalizing-flows 

HYPERION is a pipeline based on [Normalizing Flows](https://arxiv.org/abs/1912.02762) for the detection and Parameter Estimation of gravitational waves from [Close Encounters](https://arxiv.org/abs/1909.02143)

## Installation

To install hyperion you can clone this repository then install it

```
git clone https://github.com/fdesanti/HYPERION.git
cd HYPERION
python setup.py install
```

### PyTorch installation
By default (on Linux) PyTorch is shipped with the latest CUDA binaries. 
Refer to the page [PyTorch page](https://pytorch.org/get-started/locally/) for other CUDA versions and/or installation
options.

## Usage

Once the steps above have been completed hyperion can be imported and directly used. 
As an example: a typical inference code would be:

```python
from hyperion import PosteriorSampler
from hyperion.simulation.waveforms import EffectiveFlyByTemplate as efbt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#initialize the Posterior Sampler
sampler = PosteriorSampler(flow_checkpoint_path  = path_to_trained_model, 
                           waveform_generator    = efbt,
                           num_posterior_samples = 10000,
                           device                = device)

#sample posterior
posterior = sampler.sample_posterior(strain = whitened_strain,
                                     restrict_to_bounds = True)

#Once the posterior is sampled it is also possible to make corner plots
sampler.plot_corner(injection_parameters = true_parameters)


#The sampler class allows also to perform the Importance Sampling and computing Bayes Factors
resampled_posterior = sampler.reweight_posterior(posterior       = posterior,
                                                 whitened_strain = whitened_strain,
                                                 strain          = noisy_strain,
                                                 psd             = psd,
                                                 event_time      = t_gps)

print(f'The signal vs noise Bayes Factor is {sampler.BayesFactor:.2f}')
```

## Citation

If you use `HYPERION`, please cite the 
[corresponding paper](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.102004) as follows.

> De Santi et al., Physical Review D **109**, 102004 (2024)

```bibtex
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
```
`HYPERION` has also been used in:
- [arXiv:2505.02773](https://arxiv.org/abs/2505.02773)
