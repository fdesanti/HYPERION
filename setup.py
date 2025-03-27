from setuptools import setup, find_packages

__version__ = "1.0.0"

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name                          = "hyperion",
    version                       = __version__,
    description                   = "HYPERION: a fast Normalizing Flow bayesian Sampler for Gravitational Wave Observations",
    long_description              = long_description,
    long_description_content_type = 'text/markdown',
    url                           = "https://github.com/fdesanti/HYPERION",
    author                        = "Federico De Santi",
    author_email                  = "f.desanti@campus.unimib.it",
    packages                      = find_packages(),
    include_package_data          = True,
    package_data                  = {"hyperion": ["config/*"]},  
    install_requires              = [
                      #torch-related
                      "torch>=2.1.0", 
                      "torchvision", 
                      "torchaudio", 
                      "tensordict", 
                      "numpy",
                      "multiprocess",
                      #plotting and I/O
                      "seaborn", 
                      "pyyaml", 
                      "glob2", 
                      "h5py", 
                      "tqdm", 
                      #gw related
                      "astropy", 
                      "pycbc",
                      "bilby", 
                      #other
                      "scipy"],

    scripts = ['scripts/run_train.py', 
               'scripts/test_train.py', 
               'scripts/pp_plot.py'],
    
    license = "MIT",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    
    python_requires = '>=3.9',
)
