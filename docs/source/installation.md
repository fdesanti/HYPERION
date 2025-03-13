
# Installation


To install HYPERION follow these steps:

```bash
git clone https://github.com/fdesanti/HYPERION.git
cd HYPERION
python setup.py install
```

```{Note}

At the moment, HYPERION does not have a PyPy release so it cannot be installed via *pip*. You need to install it from the source as shown above.
```

## Prerequisites 

Below we list the major dependences

### PyTorch
When installing from source PyTorch will automatically detect the OS and install the latest CUDA binaries (if available).
If your machine is equipped with a GPU requiring older driver versions, refer to the page [PyTorch page](https://pytorch.org/get-started/locally/) for other CUDA versions and/or installation options.