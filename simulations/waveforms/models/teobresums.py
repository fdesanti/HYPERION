"""
Instructions for the TEOBResumS waveform model:

>>> git clone https://git.ligo.org/rossella.gamba/teobresums
>>> cd teobresums
>>> git checkout dev/DALI
>>> cd Python
>>> python setup.py install
"""

import torch
import EOBRun_module

def modes_to_k(modes):
    """
    Convert a list of modes to EOB conventions:
    k = l*(l-1)/2 + m-2;
    """
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]




class  TEOBResumSDALI():
    """Wrapper class for the TEOBResumS waveform model."""
    
    def __init__(self, fs = 2048, **kwargs):
        
        modes = [(2,1),(2,2),(3,3),(4,2)]

        
        self.default_kwargs = {
        # Initial conditions and output time grid
        'domain'             : 0,      # Time domain. EOBSPA is not available for eccentric waveforms!
        'srate_interp'       : fs,     # Srate at which to interpolate. Default = 4096.
        'use_geometric_units': "no",   # output quantities in geometric units. Default = 1
        'interp_uniform_grid': "yes",   # interpolate mode by mode on a uniform grid. Default = "no" (no interpolation)

        # Modes
        'use_mode_lm'        : modes_to_k(modes),    # List of modes to use/output through EOBRunPy.

        # ode
        'ode_tmax'           : 20e4,
        
        # nqcs
        'nqc'                : 'manual',
        'nqc_coefs_flx'      : 'none',
        'nqc_coefs_hlm'      : 'none',

        # Output parameters (Python)
        'arg_out'            : "no",     # Output hlm/hflm. Default = "no"
        'output_hpc'         : "yes",    # Output waveform. Default = 1.
        }

        self.kwargs = self.default_kwargs.update(kwargs)
        return
        

    @property
    def name(self):
        return 'TEOBResumS'
    
    @property
    def has_cuda(self):
        return False

    def __call__(self, **waveform_parameters):
        """
        Compute the waveform.
        """
        # Update the parameters
        self.kwargs.update(waveform_parameters)
        
        # Run the model
        h = EOBRun_module.EOBRunPy(self.kwargs)
        
        return h