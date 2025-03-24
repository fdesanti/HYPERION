"""
Instructions for the TEOBResumS waveform model:

>>> git clone https://git.ligo.org/waveforms/software/pyseobnr.git
>>> cd pyseobnr
>>> pip install -U pip wheel setuptools numpy
>>> pip install .
>>> pip install git+https://bitbucket.org/sergei_ossokine/waveform_tools
"""

import torch 
from importlib import import_module
from hyperion.core.utilities import HYPERION_Logger

log = HYPERION_Logger()


from pyseobnr.generate_waveform import GenerateWaveform


class PySEOBNR:
    
    
    def __init__(self, fs, **kwargs):
        try:
            pyseobnr = import_module('pyseobnr')
            self.wvf_gen = pyseobnr.generate_waveform.GenerateWaveform
        except ModuleNotFoundError as e: 
            log.error(e)
            log.warning("Unable to import pyseobnr. Please refer to the documentation to install it. PySEOBNR waveform model won't work otherwise")

        