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

from hyperion.core.distributions import q_uniform_in_components as q_prior



def modes_to_k(modes):
    """
    Convert a list of modes to EOB conventions:
    k = l*(l-1)/2 + m-2;
    """
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]


class  TEOBResumSDALI():
    """Wrapper class for the TEOBResumS waveform model."""
    
    def __init__(self, **kwargs):
        
        modes = [(2,1),(2,2),(3,3),(4,2)]

        self.default_kwargs = {
        # Initial conditions and output time grid
        'domain'             : 0,      # Time domain. EOBSPA is not available for eccentric waveforms!
        'srate_interp'       : 2048,   # Srate at which to interpolate. Default = 4096.
        'use_geometric_units': "no",   # output quantities in geometric units. Default = 1
        'interp_uniform_grid': "yes",  # interpolate mode by mode on a uniform grid. Default = "no" (no interpolation)

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
    def has_torch(self):
        return False
    
    @staticmethod
    def _check_compatability(parameters):
        """
        Check if the parameters are compatible with the model.
        """
        if all(['m1', 'm2']) in parameters.keys():
            
            m1, m2 = q_prior._sort_masses(parameters['m1'], 
                                          parameters['m2'])

            M = m1 + m2
            q = m1 / m2

            parameters['M'] = M
            parameters['q'] = q

        return parameters
    

    def __call__(self, **waveform_parameters):
        """
        Compute the waveform.
        """
        # Update the parameters       
        pars = self.kwargs
        pars.update(waveform_parameters)

        # Run the model
        eob_output = EOBRun_module.EOBRunPy(self.kwargs)
        
        output = {}
        if pars['output_hpc'] == 'yes':
            output['hp'] = eob_output[0]
            output['hc'] = eob_output[1]
        
        elif pars['arg_out'] == 'yes':
            output['t']  = torch.from_numpy(eob_output[0])
            output['hp'] = torch.from_numpy(eob_output[1])
            output['hc'] = torch.from_numpy(eob_output[2])
            output['hlm'] = torch.from_numpy(eob_output[3])
            output['dyn'] = torch.from_numpy(eob_output[4])
        
        return output