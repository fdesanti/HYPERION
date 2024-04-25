import torch
from .EFB_T_PN import PN_Expansion
from ..utilities import pc_to_Msun, Msun_to_sec

class EffectiveFlyByTemplate():
    """Class that generates templates of gw from Binary Close Encounter exploiting the Effective Fly-by Templates
     
     Args:
     -----
         fs : float
            Sampling frequency [Hz]

         duration : float
            Duration in seconds of the simulated signal. If given it will be used to build an array of times. (Default: 1)
            
         detectors : (dict, optional)
            dict of GWDetector instances into which the strain will be projected. If provided with None, only the hp and hc will be computed.

         device : str
            Device on which compute the waveforms. Either 'cpu' or 'cuda:n'. (Default: 'cpu')
            
     Methods:
     --------

        - __call__: Default method to generate the waveforms
            Args:
            -----
                m1, m2 : float
                    masses of compact objects in M_sun units. (Might have shape [Nbatch, 1])
                
                M, eta : (float, optional)
                    total mass and SYMMETRIC mass ratios in M_sun units. If provided they will be used instead of m1 and m2. (Might have shape [Nbatch, 1])
                
                distance : (float)
                    distance of the source in Mpc. (Might have shape [Nbatch, 1])
                
                p_0 : (float)
                    Semi-latus rectum of the binary system over total mass (G=c=1). (Might have shape [Nbatch, 1])
                
                e0 : (float)
                    Eccentricity at the time of periaston passage. (Might have shape [Nbatch, 1])
                
                t0_p : (float)
                    GPS time of peri-astron passage. If times_array is not given, then it will be used as central time of duration as specified in the init
                
                ra : (float)
                    right ascension of the source (in rad). (Might have shape [Nbatch, 1])
                
                dec : (float)
                    declination of the source (in rad). (Might have shape [Nbatch, 1])
                
                pol : (float)
                    polarization of the source (in rad). (Might have shape [Nbatch, 1])
                
                incl : (float)
                    inclination of the source (in rad). (Might have shape [Nbatch, 1])

                time_shift : (float)
                    Temporal shift (seconds) of the CE relative to the central strain gps_time

                times_array : (numpy.ndarray or torch.tensor)
                    Array of GPS times to use to compute the waveform. If given it will be used instead of the "duration" argument
    """
    
    def __init__(self,
                 fs = 2048, 
                 duration  = 1, 
                 detectors = None, 
                 device = 'cpu',
                 torch_compile  = False, 
                 compile_kwargs = None, 
                 ):
        
        self.fs = fs 
        self.duration  = duration
        self.detectors = detectors 
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
        
        return
    
    @property
    def name(self):
        return 'EffectiveFlyByTemplate'
    
    @property
    def has_cuda(self):
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

    @property
    def det_names(self):
        if not hasattr(self, '_det_names'):
            if self.detectors is not None:
                self._det_names= list(self.detectors.keys())
            else:
                raise ValueError("No detectors found!")
        return self._det_names
            

    """================================================================="""
    """ ==============  Define EFB-T related quantities   =============="""
    """================================================================="""
    
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
        
    '''
    def t_edge(self, l_=torch.pi):
        """
        Considering a single periastron passage, returns epoch referred to a certain eccentric anomaly value.

        Note:
        -----
        The default eccentric anomaly value is $\pi$ that corrispionds to half a lap.
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
    '''
    
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
        
        eps = 1 - ecc**2
        """
        return 1-torch.pow(self.ecc(times_array), 2)

    def semi_lat_rect(self, times_array):
        """
        Semi latus rectum of the orbit at each time.
        """
        block1 = 12.8*self.eta/torch.pow(self.p_0, 2.5)
        block2 = 1 + 0.875*torch.pow(self.e0, 2)
        return self.M*self.p_0*(1 - block1*block2*self.ecc_anomaly(times_array))
       
    """================================================================="""
    

    def get_hp_and_hc(self, times_array):
        """Function that returns the plus and cross polarization of the EFB-T template"""
        # Amplitude factors
        ampl = -(self.eta*(self.M**2))/(self.semi_lat_rect(times_array)*self.distance)
        
        #hp and hc from the PN expansion
        hp, hc = self.PN(self.phase, self.incl, self.pol, self.eps_factor)

        return torch.nan_to_num(ampl*hp), torch.nan_to_num(ampl*hc)
    
    
    def __call__(self, m1, m2, distance, p_0, e0, polarization, inclination, ra=0, dec=0, t0_p=None,time_shift = 0, M = None, eta = None, times_array=None, return_hp_and_hc=False):
        
        #TODO this is related to an issue of the GWDetector class that cannot handle multiple gps times to project
                
        if times_array is not None:
            self.duration = len(times_array) / self.fs
        else:
            times_array = torch.linspace(-self.duration/2, self.duration/2, self.duration * self.fs, dtype=torch.float64, device=self.device) #+ t0_p
            #print(times_array)

        #define physical quantities
        if m1 is not None and m2 is not None:
            self.M   = m1+m2
            self.eta = (m1*m2)/(self.M**2)
        
        
        self.distance = pc_to_Msun(distance.double()*1e6) #has to be in parsec
        
        self.e0   = e0
        self.p_0  = p_0
        self.t0_p = t0_p
        self.ra   = ra
        self.dec  = dec
        self.pol  = polarization
        self.incl = inclination
        
        # some useful quantities -------------
        self.eps0 = 1 - (e0**2) #high eccentricity parameter
        self.n0 = self._n0()
        self.Frr = self._Frr()
        
        
        self.eps_factor = self.eps(times_array)
        self.phase = self._phase(times_array) #set up phase array by passing times_array (see related func below)
        
        #compute the plus and cross polarizations
        hp, hc = self.get_hp_and_hc(times_array)
        
        #set output      
        #whether to return the polarization
        if return_hp_and_hc or self.detectors is None: 
            return {'hp': hp, 'hc': hc}

        #wether to project polarizations onto detectors
        h = {'strain': {}, 'time_delay':{}}

        for ifo in self.detectors:
            #get detector projected strain
            h['strain'][ifo]  = self.detectors[ifo].project_wave(hp, hc, ra, dec, polarization, t0_p) 

            #compute time delays with respect to Earth center
            h['time_delay'][ifo] = self.detectors[ifo].time_delay_from_earth_center(ra, dec, t0_p) + time_shift
        
        return h
    
