"""
Instructions for the PySEOBNR waveform model:

>>> git clone https://git.ligo.org/waveforms/software/pyseobnr.git
>>> cd pyseobnr
>>> pip install -U pip wheel setuptools numpy
>>> pip install .
>>> pip install git+https://bitbucket.org/sergei_ossokine/waveform_tools
"""

import torch 
from importlib import import_module
from ....core.utilities import HYPERION_Logger

log = HYPERION_Logger()

class PySEOBNR:
    """
    Wrapper class for the PySEOBNR waveform model.
  
    Args:
        fs (float): Sampling frequency of the waveform.
        kwargs    : Additional keyword arguments to pass to the EOBRun_module generator.

    By default, the waveform model is set to use the following parameters:
        - **modes**             : [(2, 2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4)]  hlm modes of the waveform to use
        - **approximant**       : "SEOBNRv5EHM"                                     the approximant to use: Either "SEOBNRv5EHM" (eccentric) or "SEOBNRv5PHM" (precessing). (Default: "SEOBNRv5EHM")
        - **f22_start**         : 10                                                starting frequency [Hz] of the (2,2) mode
        - **deltaT**            : 1.0/fs                                            delta T
        - **f_max**             : None                                              maximum frequency of the waveform
        - **settings**          : {}                                                other settings for the waveform generation
        - **EccIC**             : 0                                                 EccIC = 0 for instantaneous initial orbital frequency, and EccIC = 1 for orbit-averaged initial orbital frequency
    """

    def __init__(self, fs, **kwargs):
        try:
            pyseobnr = import_module('pyseobnr')
            self.GenerateWaveform = pyseobnr.generate_waveform.GenerateWaveform
        except ModuleNotFoundError as e: 
            log.error(e)
            log.warning("Unable to import pyseobnr. Please refer to the documentation to install it. PySEOBNR waveform model won't work otherwise")

        self.fs = fs

        self.default_kwargs = {
            "modes"      : [(2, 2), (2, 1), (3, 2), (3, 3), (4, 3), (4, 4)], #hlm modes of the waveform to use
            "approximant": "SEOBNRv5EHM",                                    #the approximant to use: Either "SEOBNRv5EHM" (eccentric) or "SEOBNRv5PHM" (precessing). (Default: "SEOBNRv5EHM")
            "f22_start"  : 10,                                               #starting frequency of the (2,2) mode
            "deltaT"     : 1.0/fs,                                           #delta T
            "f_max"      : None,                                             #maximum frequency of the waveform
            "settings"   : {},                                               #settings for the waveform generation
            "EccIC"      : 0,                                                # EccIC = 0 for instantaneous initial orbital frequency, and EccIC = 1 for orbit-averaged initial orbital frequency
        }

        self.kwargs = self.default_kwargs.copy()
        if kwargs:
            self.kwargs.update(kwargs) 

        print('\n')
        log.info(f'Using {self.name} waveform model with the following parameters:')
        for key, value in self.kwargs.items():
            print(f'{key}: {value}')
        print('\n')

    @property
    def name(self):
        return self.default_kwargs["approximant"]
    
    @property
    def has_torch(self):
        return False
    
    @staticmethod
    def _validate_parameters(parameters):
        """
        If the parameters contains single masses and convert them to
        total mass M and mass ratio q.
        If parameters contains luminosity_distance as a key we convert it to distance.
        """
        #check masses 
        if all(p in parameters.keys() for p in ['m1', 'm2']):      #change m1, m2 to mass_1, mass_2
            m2, m1 = sorted([parameters['m1'], parameters['m2']])
            parameters['mass1'] = m1
            parameters['mass2'] = m2
            parameters.pop('m1')
            parameters.pop('m2')

        elif all(p in parameters.keys() for p in ['M', 'q']):      #change M, q to mass_1, mass_2
            M = parameters['M']
            q = parameters['q']
            parameters['mass1'] = M / (1 + q)
            parameters['mass2'] = M * q / (1 + q)
            parameters.pop('M')
            parameters.pop('q')

        elif all(p in parameters.keys() for p in ['Mchirp', 'q']): #change Mchirp, q to mass_1, mass_2
            Mchirp = parameters['Mchirp']
            q = parameters['q']
            M = Mchirp * (q / (1 + q)**2)**(-3/5)
            parameters['mass1'] = M / (1 + q)
            parameters['mass2'] = M * q / (1 + q)
            parameters.pop('Mchirp')
            parameters.pop('q')

        #check luminosity distance  
        if 'luminosity_distance' in parameters.keys():
            parameters['distance'] = parameters.pop('luminosity_distance')
        
        return parameters
    
    def __call__(self, waveform_parameters):
        """
        Computes the waveform.

        Args:
            waveform_parameters (TensorSamples): Dictionary containing the waveform parameters.

        Returns:
            dict: Dictionary containing the waveform data

            -   **t**  (torch.Tensor): Time array of the output waveform
            -   **hp** (torch.Tensor): Plus polarization waveform
            -   **hc** (torch.Tensor): Cross polarization waveform
        """
        # Update the parameters      
        pars = self.kwargs.copy()

        #check parameter consistency
        waveform_parameters = self._validate_parameters(waveform_parameters)
        pars.update(waveform_parameters)
       
        # Run the waveform generator
        wfm_gen = self.GenerateWaveform(params_dict) 
        hp, hc  = wfm_gen.generate_td_polarizations()
        times   = hp.deltaT * np.arange(hp.data.length) + hp.epoch.ns() * 1e-9
        
        #set the output
        output = {}
        output['t']  = torch.from_numpy(times)
        output['hp'] = torch.from_numpy(hp.data.data)
        output['hc'] = torch.from_numpy(hc.data.data)

        return output