import torch
from .EFB_T_PN import PN_Expansion
from ..waveform_utilities import pc_to_Msun, Msun_to_sec

class EffectiveFlyByTemplate():
    """
    Class that generates templates of gw from Binary Close Encounter exploiting the Effective Fly-by Templates. 
    (See  N. Loutrel 2019 for details - arXiv:1909.02143)
     
    Args:
     
        fs            (float): Sampling frequency of the waveform in Hz. (Default is 2048)
        duration      (float): Duration of the waveform in seconds. (Default is 1)
        device          (str): Device to use for the computation. (Default is 'cpu')
        torch_compile  (bool): Whether to compile the PN expansion with torch.jit. (Default is False)
        compile_kwargs (dict): Additional keyword arguments to pass to the torch.compile function. (Default is None)
    """
    def __init__(self,
                 fs             = 2048,
                 duration       = 1,
                 device         = 'cpu',
                 torch_compile  = False,
                 compile_kwargs = None,
                 ):
        
        self.fs = fs 
        self.duration  = duration
        self.device    = device

        self.nmax = 3
        self.kmax = 7
        
        #setting up the model of PN expansion
        _compile_kwargs = {'mode':'max-autotune-no-cudagraphs', 
                           'disable': False if torch_compile else True}
        if compile_kwargs is not None:
            _compile_kwargs.update(compile_kwargs)
        
        #getting the PN expansion object
        self.PN = torch.compile(PN_Expansion(nmax=3, kmax=7), **_compile_kwargs)
        
    
    @property
    def name(self):
        return 'EffectiveFlyByTemplate'
    
    @property
    def has_torch(self):
        return True
    
    @property
    def fs(self):
        return self._fs
    @fs.setter
    def fs(self, value):
        self._fs = value

    @property
    def duration(self):
        return self._duration
    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def f_source(self):
        return 1/self.duration

    #================================================================
    #==============  Define EFB-T related quantities   ==============
    #================================================================
    def _n0(self):
        """returns the n0 parameter entering the EFB-T expansion"""
        nblock_1 = 1/self.M
        nblock_2 = (self.eps0/self.p_0)**(3/2)
        return 1/Msun_to_sec(1/(nblock_1*nblock_2))
        
    def _Frr(self):
        """returns the Frr parameter entering the EFB-t expansion"""
        fblock_1 = 96/(10*torch.pi) 
        fblock_2 = self.eta/(self.M*(self.p_0**4))
        fblock_3 = torch.sqrt(1-self.e0**2)
        fblock_4 = 1 + (73/24)*(self.e0**2) + (37/96)*(self.e0**4)
        return 1/Msun_to_sec(1/(fblock_1*fblock_2*fblock_3*fblock_4))
        
    def _phase(self, times_array):
        """
        Phase of the first body at each time.
        """
        sqrt_epst = torch.sqrt(self.eps(times_array))        
        #print('ecc_anomlay', (self.ecc(times_array))[0])
        return self.ecc_anomaly(times_array)/(torch.log((1+sqrt_epst)/self.ecc(times_array))-sqrt_epst)
        
    
    def t_edge(self, l_=torch.pi):
        """
        Considering a single periastron passage, returns epoch referred to a certain eccentric anomaly value.

        Note:
            The default eccentric anomaly value is  that corrispionds to half a lap.
        """
        assert l_!=0
        lblock_1 = torch.log(((2*torch.pi*l_*self.Frr)/self.n0) + 1) 
        if torch.isnan(lblock_1):
            if l_<0:
                return self.t0_p - 30 
            if l_>0:
                return self.t0_p + 30             
        lblock_2 = 2*torch.pi*self.Frr
        return (lblock_1/lblock_2) + self.t0_p
    
    
    def ecc_anomaly(self, times_array):
        """
        Eccentric anomaly of the first body at each time.
        """
        block1 = self.n0/(2*torch.pi*self.Frr)      
        block2 = torch.subtract(torch.exp(2*torch.pi*self.Frr*times_array), 1)
        #print(self.Frr[0])
        #print('ecc_anomaly', (block2.max()))

        return block1*block2

    # How orbital parameters change with time during pericenter passage
    def ecc(self, times_array):
        """
        Eccentricity of the orbit at each time.
        """
        #block1 = 20.267*self.eta*self.e0/self.p_0**2.5
        #block2 = 1 + 0.39803*self.e0**2
        block1 = (304/15)*self.eta*self.e0/self.p_0**(5/2)
        block2 = 1 + (121/304)*self.e0**2
        
        #print((block1*block2)[0])
        return self.e0 - block1*block2*self.ecc_anomaly(times_array)

    def eps(self, times_array):
        """
        High eccentricity parameter at each time.
        
        .. math::
            \epsilon = 1 - e(t)^2
        """
        return 1-torch.pow(self.ecc(times_array), 2)

    def semi_lat_rect(self, times_array):
        """
        Semi latus rectum of the orbit at each time.
        """
        block1 = 12.8*self.eta/torch.pow(self.p_0, 2.5)
        block2 = 1 + 0.875*torch.pow(self.e0, 2)
        return self.M*self.p_0*(1 - block1*block2*self.ecc_anomaly(times_array))
           

    def get_hp_and_hc(self, times_array):
        """
        Returns the plus and cross polarization of the EFB-T template
        """
        # Amplitude factors
        ampl = -(self.eta*(self.M**2))/(self.semi_lat_rect(times_array)*self.distance)
        
        #hp and hc from the PN expansion
        hp, hc = self.PN(self.phase, self.incl, self.pol, self.eps_factor)

        return torch.nan_to_num(ampl*hp), torch.nan_to_num(ampl*hc)
    
    @staticmethod
    def _check_parameters(parameters):
        """
        If the parameters contains single masses and convert them to
        total mass M and mass ratio q.
        If parameters contains luminosity_distance as a key we convert it to distance.
        """
        #check masses 
        if all(p in parameters.keys() for p in ['m1', 'm2']):
            m2, m1 = [parameters['m1'], parameters['m2']]
            parameters['M']   = m1 + m2
            parameters['eta'] = (m1*m2)/(M**2)

        elif all(p in parameters.keys() for p in ['M', 'q']):
            q = parameters['q']
            eta = q / (1+q)**2
            parameters['eta'] = eta
        
        elif all(p in parameters.keys() for p in ['Mchirp', 'q']):
            Mchirp, q = [parameters['Mchirp'], parameters['q']]
            eta = q / (1+q)**2
            parameters['M']   = Mchirp * eta**(-3/5)
            parameters['eta'] = eta

        #check luminosity distance  
        if 'luminosity_distance' in parameters.keys():
            parameters['distance'] = parameters.pop('luminosity_distance')
        
        return parameters
    
    
    def __call__(self, waveform_parameters):
        """
        Generate the waveform of the EFB-T model.

        Args:
            waveform_parameters (TensorSamples): Dictionary that must contain the following parameters:

                - **M**            : Total mass of the binary system in M_sun units.

                - **eta**          : Symmetric mass ratio of the binary system.

                - **m1** (optional): Mass of the first compact object in M_sun units. If provided, it will be used to compute M and eta

                - **m2** (optional): Mass of the second compact object in M_sun units. If provided, it will be used to compute M and eta

                - **distance**     : Luminosity distance of the source in Mpc.

                - **p_0**          : Semi-latus rectum of the binary system over total mass (G=c=1).

                - **e0**           : Eccentricity at the time of periaston passage.

                - **t0_p**         : GPS time of peri-astron passage.

                - **polarization** : Polarization of the source (in rad).   

        Returns:
            tuple: A tuple containing:

                - **hp** (torch.Tensor): Plus polarization waveform.

                - **hc** (torch.Tensor): Cross polarization waveform.

                - **t0_p** (torch.Tensor): Time of peri-astron passage.

        Caution:
            Opposite to other waveform models, the EFB-T requires polarization angle to be treated as an *intrinsic* parameter.
            Make sure to provide the same polarization angle when projecting the waveform onto the detectors.

        Hint:
            You can provide the same polarization angle by setting the same ``seed`` value when defining the *intrinsics* and *extrinsics* prior for this parameter.
    """
        #define the times array       
        times_array = torch.linspace(-self.duration/2, self.duration/2, self.duration * self.fs, dtype=torch.float64, device=self.device) #+ t0_p
    
        #check parameters consistency
        waveform_parameters = self._check_parameters(waveform_parameters)

        #extract parameters
        self.M    = waveform_parameters['M']
        self.eta  = waveform_parameters['eta']
        distance  = waveform_parameters['distance']
        self.e0   = waveform_parameters['e0']
        self.p_0  = waveform_parameters['p_0']
        self.t0_p = waveform_parameters['t0_p']
        self.ra   = waveform_parameters['ra']
        self.dec  = waveform_parameters['dec']
        self.pol  = waveform_parameters['polarization']
        self.incl = waveform_parameters['inclination']
        
        #convert distance to Msun
        self.distance = pc_to_Msun(distance.double()*1e6) #has to be in parsec
        
        # some useful quantities -------------
        self.eps0 = 1 - (e0**2) #high eccentricity parameter
        self.n0   = self._n0()
        self.Frr  = self._Frr()
        
        self.eps_factor = self.eps(times_array)
        self.phase = self._phase(times_array) #set up phase array by passing times_array (see related func below)
        
        #compute the plus and cross polarizations
        hp, hc = self.get_hp_and_hc(times_array)
        
        #set output      
        return hp, hc, t0_p
    
