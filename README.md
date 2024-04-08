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
Add these lines to your /.bash_profile (/.zprofile on Mac)

```
PYTHONPATH = "path_to_work_dir":$PYTHONPATH
export PYTHONPATH
```

then run 

```
source ~/.bash_profile
```

### A note about PyTorch
By default PyTorch will download (on Linux) the latest CUDA binaries (12.1). 
Refer to the page https://pytorch.org/get-started/locally/ for other CUDA versions
(e.g. PcUniverse only works with CUDA <= 11.8)

## Usage



