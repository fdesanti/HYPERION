"""
Instructions for the TEOBResumS waveform model:

>>> git clone https://git.ligo.org/rossella.gamba/teobresums
>>> cd teobresums
>>> git checkout dev/DALI
>>> cd Python
>>> python setup.py install
"""

import torch
from importlib import import_module
from hyperion.core.utilities import HYPERION_Logger

log = HYPERION_Logger()

'''
try:
    import EOBRun_module
except ModuleNotFoundError as e: 
    log.error(e)
    log.warning("Unable to import EOBRun_module. Please refer to the documentation to install it. TEOBResumSDALI waveform model won't work otherwise")
'''


def modes_to_k(modes):
    """
    Convert a list of modes to EOB conventions:
    k = l*(l-1)/2 + m-2;
    """
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]


class  TEOBResumSDALI():
    """
    Wrapper class for the TEOBResumS waveform model.
    
    Args:
    -----
        fs (float): Sampling frequency of the waveform.
        **kwargs: Additional keyword arguments to pass to the EOBRun_module generator.
    """
    
    def __init__(self, fs, **kwargs):
        try:
        #import EOBRun_module
            eob_module = import_module('EOBRun_module')
            self.EOBRunPy = eob_module.EOBRunPy
        except ModuleNotFoundError as e: 
            log.error(e)
            log.warning("Unable to import EOBRun_module. Please refer to the documentation to install it. TEOBResumSDALI waveform model won't work otherwise")

        
        modes = [(2,1),(2,2),(3,3),(4,2)]

        self.default_kwargs = {
        # Initial conditions and output time grid
        'domain'             : 0,      # Time domain. EOBSPA is not available for eccentric waveforms!
        'srate_interp'       : float(fs),   # Srate at which to interpolate. Default = 4096.
        'use_geometric_units': "no",   # output quantities in geometric units. Default = 1
        'interp_uniform_grid': "yes",  # interpolate mode by mode on a uniform grid. Default = "no" (no interpolation)
        'initial_frequency'  : .02,     # in Hz if use_geometric_units = 0, else in geometric units
        'ecc_freq'           : 1,      # Use periastron (0), average (1) or apastron (2) frequency for initial condition computation. Default = 1

        # Modes
        'use_mode_lm'        : modes_to_k(modes), # List of modes to use/output through EOBRunPy.

        # ode
        'ode_tmax'           : 20e4,
        'ode_tstep_opt'      : "adaptive",        # Fixing uniform or adaptive. Default = 1 
        
        # nqcs
        #'nqc'                : 'manual',           #options are ["none", "manual", "auto"]
        #'nqc_coefs_flx'      : 'nrfit_spin202002', #options are ["none", "nrfit_spin202002", "nrfit_nospin201602"]
        #'nqc_coefs_hlm'      : 'nrfit_spin202002', #options are ["none", "nrfit_spin202002", "nrfit_nospin201602", "compute"]

        # Output parameters (Python)
        'arg_out'            : "no",     # Output hlm/hflm. Default = "no"
        'output_hpc'         : "no",     # Output waveform. Default = 1.
        }
        
        self.kwargs = self.default_kwargs.copy()
        if kwargs:
            self.kwargs.update(kwargs) 
            
        print('\n')
        log.info('Using TEOBResumS-Dali waveform model with the following parameters:')
        for key, value in self.kwargs.items():
            print(f'{key}: {value}')
        print('\n')
        #print(self.kwargs, '\n')
        return
        
    @property
    def name(self):
        return 'TEOBResumS'
    
    @property
    def has_torch(self):
        return False
    
    @property
    def fs(self):
        return self.kwargs['srate_interp']
    
    
    @staticmethod
    def _check_masses(parameters):
        """
        Check if the parameters contains single masses and convert them to
        total mass M and mass ratio q.
        """
        if all(p in parameters.keys() for p in ['m1', 'm2']):
            
            m2, m1 = sorted([parameters['m1'], parameters['m2']])

            parameters['M'] = m1 + m2
            parameters['q'] = m1 / m2
            parameters.pop('m1')
            parameters.pop('m2')
    
        return parameters
    

    def __call__(self, waveform_parameters):
        """
        Compute the waveform.
        """
        # Update the parameters      
        pars = self.kwargs.copy()

        waveform_parameters = self._check_masses(waveform_parameters)
        pars.update(waveform_parameters)
       
        # Run the model
        eob_output = self.EOBRunPy(pars)
        
        output = {}
        
        output['t']  = torch.from_numpy(eob_output[0])
        output['hp'] = torch.from_numpy(eob_output[1])
        output['hc'] = torch.from_numpy(eob_output[2])
        
        if pars['arg_out'] == 'yes':
            output['hlm'] = torch.from_numpy(eob_output[3])
            output['dyn'] = torch.from_numpy(eob_output[4])
        return output