import torch
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from pycbc.waveform import get_td_waveform, get_fd_waveform

wvf_gen = {'TD': get_td_waveform, 'FD': get_fd_waveform}

class PyCBCWaveform:
    """
    Wrapper class to the PyCBC waveform models.

    Args:
    -----
        fs (float): sampling frequency
        mode (str): 'TD' or 'FD'. (Default: 'TD')
        **kwargs: Additional keyword arguments to pass to the waveform generator
    """

    def __init__(self, fs=2048, mode='TD', **kwargs):
        #set mode
        self.mode = mode

        #we let pycbc handle the exceptions
        self.kwargs = kwargs

        if mode == 'TD':
            self.kwargs['delta_t'] = 1/fs
        

    @property
    def name(self):
        return self.mode
    
    @property
    def has_torch(self):
        return False
    
    @staticmethod
    def _check_parameters(parameters):
        """
        If the parameters contains single masses and convert them to
        total mass M and mass ratio q.
        If parameters contains luminosity_distance as a key we convert it to distance.
        """
        #check masses 
        if all(p in parameters.keys() for p in ['m1', 'm2']):      #change m1, m2 to mass_1, mass_2
            m2, m1 = sorted([parameters['m1'], parameters['m2']])
            parameters['mass_1'] = m1
            parameters['mass_2'] = m2
            parameters.pop('m1')
            parameters.pop('m2')

        elif all(p in parameters.keys() for p in ['M', 'q']):      #change M, q to mass_1, mass_2
            M = parameters['M']
            q = parameters['q']
            parameters['mass_1'] = M / (1 + q)
            parameters['mass_2'] = M * q / (1 + q)
            parameters.pop('M')
            parameters.pop('q')

        elif all(p in parameters.keys() for p in ['Mchirp', 'q']): #change Mchirp, q to mass_1, mass_2
            Mchirp = parameters['Mchirp']
            q = parameters['q']
            M = Mchirp * (q / (1 + q)**2)**(-3/5)
            parameters['mass_1'] = M / (1 + q)
            parameters['mass_2'] = M * q / (1 + q)
            parameters.pop('Mchirp')
            parameters.pop('q')

        #check luminosity distance  
        if 'luminosity_distance' in parameters.keys():
            parameters['distance'] = parameters.pop('luminosity_distance')
        
        return parameters
    
    def __call__(self, waveform_parameters):
        """
        Compute the waveform.
        """
        # Update the parameters      
        pars = self.kwargs.copy()

        #check parameter consistency
        waveform_parameters = self._check_parameters(waveform_parameters)
        pars.update(waveform_parameters)

        # Run the model
        hp, hc = wvf_gen[self.mode](**pars)
        
        #set the output
        output = {}
        output['hp'] = torch.from_numpy(hp)
        output['hc'] = torch.from_numpy(hc)
        return output