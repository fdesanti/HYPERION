from setuptools import setup, find_packages

setup(
    name = "hyperion",  
    version = "0.1.0",  
    description = "HYPERION: a fast Normalizing Flow bayesian Sampler for Gravitational Wave Observations",  
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',  
    url = "https://github.com/fdesanti/HYPERION",  
    author = "Federico De Santi",
    author_email = "f.desanti@studenti.unipi.it",
    license = "MIT", 
    packages = find_packages(), 
    install_requires=["torch", 
                      "torchvision", 
                      "torchaudio", 
                      "tensordict", 
                      "numpy",
                      "seaborn", 
                      "pyyaml", 
                      "glob2", 
                      "h5py", 
                      "tqdm", 
                      "astropy", 
                      "bilby"],  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8', 
)
