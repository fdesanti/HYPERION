import yaml

from tensordict import TensorDict

from importlib import import_module

from astropy import units as u
from astropy.time import Time
from astropy.constants import R_earth, c
from astropy.coordinates import EarthLocation

R_earth = R_earth.value #earth radius value [m]
c = c.value #speed of light value [m/s]

from ..config import CONF_DIR
from ..core.utilities import HYPERION_Logger

log = HYPERION_Logger()

def get_detectors_configs(det_conf_path = None):
    #get the detectors configuration parameters
    #try:
    if not det_conf_path:
        det_conf_path = f"{CONF_DIR}/detectors.yml"
    with open(det_conf_path, 'r') as file:
        det_configs = yaml.safe_load(file)    
    
    # except (FileNotFoundError, OSError) as e:
    #     log.error(e)
    #     log.error(f"Detector configuration file not found at {det_conf_path}")
    #     det_configs = dict()
        
    return det_configs

detectors_configs = get_detectors_configs()
available_detectors = list(detectors_configs.keys())


class GWDetector(object):
    """
    Class for a gravitational wave detector

    Arguments:
    ----------
        name : string
            Name of the detector. Use list_available_detectors() to see a list of available names. Must be set to None to use a custom detector.
            If None and config_file_path is provided, it is asserted directly from the configuration file
            
        reference_time (GPS): float
            Reference GPS time at which the detector is initialized
            
        use_torch : bool
            Whether to use torch or numpy. "use_torch = True" is useful to exploit code parallelization, optionally on GPU.
            (Default: False)
            
        device : str ('cpu'/'gpu')
            The device on which the class operates. It matters only when torch is used. (Default: 'cpu')
            
        config_file_path : str
            Path to a custom detector configuration file. "name" must be set to None in order for that file to be used.
    """

    def __init__(self, name = None, reference_time = 1370692818, use_torch=False, device = 'cpu', config_file_path = None):
        """Constructor"""

        if device == 'cuda':
            try:
                assert use_torch == True, "Cannot use GPU (cuda) without using torch"
            except AssertionError as e:
                log.warning(e)
                log.info('Setting use_torch to True')
                use_torch = True
        
        self.device    = device
        
        if use_torch:
            self.xp = import_module('torch')
            from ..core.utilities import interp1d
            self.interp = interp1d     
        else:
            self.xp = import_module('numpy')
            self.interp = self.xp.interp
        self.use_torch = use_torch
        

        #loading configuration parameters --------------------------------------------
        if name is not None:
            assert name in available_detectors, f"{name} detector not available. Available ones are {available_detectors}. \
                                                 Please select one of those or provide a custom config file path"
            conf_params = detectors_configs[name]
            self.name = name
        else:
            assert config_file_path is not None, "No name specified, please provide at least a custom detector config file"
            detector = get_detectors_configs(config_file_path)
            self.name = list(detector.keys())[0] #get the detector name from the configuration file
            conf_params = detector[self.name]
            
        self.latitude               = conf_params['latitude']
        self.longitude              = conf_params['longitude']
        self.elevation              = conf_params['elevation']
        self.angle_between_arms     = conf_params['angle_between_arms']
        
        if 'arms_orientation' in conf_params:
            self.arms_orientation_angle = conf_params['arms_orientation']
            
        elif ('xarm_azimuth' in conf_params) and ('yarm_azimuth' in conf_params):
            self.arms_orientation_angle = 0.5*(conf_params['xarm_azimuth'] + conf_params['yarm_azimuth'])
    
        
        self.reference_time = reference_time
        return
    
    @property
    def reference_time(self):
        return self._reference_time
    @reference_time.setter
    def reference_time(self, value):
        self._reference_time = value
    
    #LATITUDE -----------------------------------------
    @property
    def latitude(self):
        return self._latitude
    @latitude.setter
    def latitude(self, value):
        if self.use_torch:
            value = self.xp.tensor(value)
        self._latitude = self.xp.deg2rad(value)
        
    #LONGITUDE ----------------------------------------
    @property
    def longitude(self):
        return self._longitude
    @longitude.setter
    def longitude(self, value):
        if self.use_torch:
            value = self.xp.tensor(value)
        self._longitude = self.xp.deg2rad(value)
        
    #ELEVATION ---------------------------------------
    @property
    def elevation(self):
        return self._elevation
    @elevation.setter
    def elevation(self, value):
        if self.use_torch:
            value = self.xp.tensor(value)
        self._elevation = self.xp.deg2rad(value)
        
    #ARMS ORIENTATION ---------------------------------
    @property
    def arms_orientation_angle(self):
        return self._arms_orientation_angle
    @arms_orientation_angle.setter
    def arms_orientation_angle(self, value):
        if self.use_torch:
            value = self.xp.tensor(value)
        self._arms_orientation_angle = self.xp.deg2rad(value)
        
    #ANGLE BETWEEN ARMS--------------------------------
    @property
    def angle_between_arms(self):
        return self._angle_between_arms
    @angle_between_arms.setter
    def angle_between_arms(self, value):
        if self.use_torch:
            value = self.xp.tensor(value)
        self._angle_between_arms = self.xp.deg2rad(value)
        
        
    #GEOCENTRIC LOCATION ------------------------------
    @property
    def location(self):
        if hasattr(self, "_location"):
            return self._location 
        else:
            earthloc = EarthLocation.from_geodetic(float(self.longitude) * u.rad, 
                                                   float(self.latitude)  * u.rad,
                                                   float(self.elevation) * u.meter)
            x, y, z = earthloc.x.value, earthloc.y.value, earthloc.z.value
            loc = [x, y, z]
            if self.use_torch:
                self._location = self.xp.tensor(loc).to(self.device)
            else:
                self._location = self.xp.array(loc)
            return self._location
    
    
    def lst(self, t_gps):
        """returns the local sidereal time at the observatory site."""        
        return self.lst_estimate(t_gps, sidereal_time_kwargs={'kind':'mean'})
    
    def gmst(self, t_gps):
        """returns the Greenwich Mean Sidereal Time."""
        return self.lst_estimate(t_gps, sidereal_time_kwargs={'kind':'apparent', 'longitude': 'greenwich'})
          
          
    def lst_estimate(self, t_gps, sidereal_time_kwargs):
        """
        Estimate the local apparent sidereal time at the observatory site. This is necessary fit the detector arms
        relative rotation over the observation time.

        Args:
        -----
            param t_gps: float, int, list, numpy.ndarray or torch.tensor
                time of arrival of the source signal.
        Return:
        -------
            lst: float or numpy.ndarray or torch.tensor 
                local sidereal time(s) in rad. 
        """
        
        if self.use_torch:
            if  self.xp.is_tensor(t_gps):
                t_gps = t_gps.cpu().numpy() #convert to numpy array since astropy cannot accept tensors. The cpu() options is a
                                                      #precautionary measure in case "t_gps" lies on a gpu
        
        lst = Time(t_gps, format='gps', scale='utc',
                         location=('{}d'.format(self.xp.rad2deg(self.longitude)), 
                                   '{}d'.format(self.xp.rad2deg(self.latitude)))).sidereal_time(**sidereal_time_kwargs).rad
        
        if self.use_torch:
            lst = self.xp.tensor(lst).to(self.device) #convert back to tensor and move on its original device (either cpu or gpu)
        return lst
    
    
    def _ab_factors(self, xangle, ra, dec, t_gps):
        """
        Method that calculates the amplitude factors of plus and cross
        polarizationarization in the wave projection on the detector.
        (See: Phys. Rev. D 58, 063001)
            
        Args:
        -----
            xangle: float
                The orientation of the detector's arms with respect to local geographical direction, in
                rad. It is measured counterclock-wise from East to the bisector of the interferometer arms

            ra: float
                Right ascension of the source in rad.
                
            dec: float
                Declination of the source in rad.
                
            t_gps: float or ndarray or torch.tensor
                GPS time at which compute the detector response function
         
         Returns:
         --------
            a, b: tuple of float or numpy.ndarray or torch.tensors
                relative amplitudes of hplus and hcross.
        """
        
        #get lst
        lst = self.lst(t_gps)
        
        a = (1/16)*self.xp.sin(2*xangle)*(3-self.xp.cos(2*self.latitude))*(3-self.xp.cos(2*dec))*self.xp.cos(2*(ra - lst))-\
             (1/4)*self.xp.cos(2*xangle)*self.xp.sin(self.latitude)*(3-self.xp.cos(2*dec))*self.xp.sin(2*(ra - lst))+\
             (1/4)*self.xp.sin(2*xangle)*self.xp.sin(2*self.latitude)*self.xp.sin(2*dec)*self.xp.cos(ra - lst)-\
             (1/2)*self.xp.cos(2*xangle)*self.xp.cos(self.latitude)*self.xp.sin(2*dec)*self.xp.sin(ra - lst)+\
             (3/4)*self.xp.sin(2*xangle)*(self.xp.cos(self.latitude)**2)*(self.xp.cos(dec)**2)

        b = self.xp.cos(2*xangle)*self.xp.sin(self.latitude)*self.xp.sin(dec)*self.xp.cos(2*(ra - lst))+\
             (1/4)*self.xp.sin(2*xangle)*(3-self.xp.cos(2*self.latitude))*self.xp.sin(dec)*self.xp.sin(2*(ra - lst))+\
             self.xp.cos(2*xangle)*self.xp.cos(self.latitude)*self.xp.cos(dec)*self.xp.cos(ra - lst)+\
             (1/2)*self.xp.sin(2*xangle)*self.xp.sin(2*self.latitude)*self.xp.cos(dec)*self.xp.sin(ra - lst)

        return a, b

    def antenna_pattern_functions(self, ra, dec, polarization, t_gps):
        '''
        Evaluate the antenna pattern functions.

        Args:
        -----
            ra: float
                Right ascension of the source in rad.

            dec: float
                Declination of the source in rad.

            polarization: float
                polarizationarization angle of the wave in rad.

            t_gps: float, int, list , numpy.ndarray or torch.tensor
                time of arrival of the source signal.

        Returns:
        --------
            fplus, fcross: tuple of float or numpy.ndarray or torch.tensors
                Antenna pattern response functions for the plus and cross polarizations respectively
        '''
        
        xangle = self.arms_orientation_angle    
        
        ampl11, ampl12 = self._ab_factors(xangle, ra, dec, t_gps)
        
        fplus  = self.xp.sin(self.angle_between_arms)*(ampl11*self.xp.cos(2*polarization) + ampl12*self.xp.sin(2*polarization))
        fcross = self.xp.sin(self.angle_between_arms)*(ampl12*self.xp.cos(2*polarization) - ampl11*self.xp.sin(2*polarization))

        return fplus, fcross

    def project_wave(self, hp, hc, ra, dec, polarization, t_gps=None):
        """
        Project the plus and cross gw polarizations into the detector frame

        Args:
        -----
            hp, hc: array, tensor or pycbc/gwpy TimeSeries
                Plus and cross polarizations of the wave.

            ra: float, array, tensor
                Right ascension of the source in rad.

            dec: float, array, tensor
                Declination of the source in rad.

            polarization: float, array, tensor
                polarizationarization angle of the wave in rad.
                
            t_gps: float, array, tensor
                GPS time of the event
                
        Returns:
        --------
            h: either numpy.ndarray or torch.tensor or pycbc/gwpy TimeSeries
                Final gravitational wave signal (detector stain).
        """
        #assert(len(hp)==len(hc))
        #t_gpss = hp.get_sampled_time()
        if t_gps is None:
            t_gps = self.reference_time
        
        if self.use_torch: 
            #check device and move to the right one 
            hp           = hp.to(self.device)
            hc           = hc.to(self.device)
            ra           = ra.to(self.device)
            dec          = dec.to(self.device)
            polarization = polarization.to(self.device)
            
        f_plus, f_cross = self.antenna_pattern_functions(ra, dec, polarization, t_gps)

        h_signal = hp*f_plus + hc*f_cross
        
        return h_signal
    
    def time_delay_from_earth_center(self, ra, dec, t_gps=None):
        """Returns the time delay from Earth Center"""
        #define earth center on right device (if torch is used)
        if self.use_torch:
            ra  = ra.to(self.device)
            dec = dec.to(self.device)
            kw  = {'device':self.device}
        else: 
            kw = {}
        earth_center = self.xp.zeros(3, **kw)
        return self.time_delay_from_location(earth_center, ra, dec, t_gps)
    
    
    def time_delay_from_location(self, other_location, ra, dec, t_gps = None):
        """
        Return the time delay from the given location to detector for
        a signal from a certain sky location at given GPS time(s).
        It supports batched computation with either numpy arrays or torch tensors.
        Adapted from PyCBC and Lalsimulation.
        
        Args:
        -----
            other_location : list, array, tensor or GWDetector instance
                Earth Geocenter coordinates or GWDetector instance
                
            ra : float, array, tensor
                The right ascension (in rad) of the signal.
                
            declination : float, array, tensor
                The declination (in rad) of the signal.
                
            t_gps : float, array, tensor
                The GPS time (in s) of the signal. If None, the reference time of the detector is used.

        Returns:
        --------
            dt : float, array, tensor
                The arrival time difference between the detectors.
        """

        if t_gps is None:
            t_gps = self.reference_time
        
        if isinstance(other_location, GWDetector):
            other_location = other_location.location
        
        #compute relative position vector
        dx = other_location - self.location
        
        #greenwich hour angle
        gha = self.gmst(t_gps) - ra 
    
        #compute the components of unit vector pointing from the geocenter to the surce
        e0 = self.xp.cos(dec) * self.xp.cos(gha)
        e1 = self.xp.cos(dec) * -self.xp.sin(gha)
        e2 = self.xp.sin(dec)
        
        #compute the dot product 
        #we do it manually to enable batched computation / multi gps times
        dt = (dx[0]*e0 + dx[1]*e1 + dx[2]*e2) / c
        return dt


###########################################
#---------- Detector Network --------------
###########################################    
class GWDetectorNetwork():
    """
    Class for a network of gravitational wave detectors.
    It accepts a list of detector names (or a list of GWDetector instances).
    The class allows for the gw projection on the detector network.


    Arguments:
    ----------
        names : list of strings
            List of detector names. Use list_available_detectors() to see a list of available names. 
            Mutually exclusive with <detectors> argument.

        detectors : dict of GWDetector instances
            List of GWDetector instances. If it is provided, <names> will be ignored.

        kwargs : dict
           Optional initialization parameters for the detectors. If <detectors> is provided, this argument is ignored.
           See the documentation of the GWDetector class for more details.

    """

    def __init__(self, names=['H1', 'L1', 'V1'], detectors=None, **kwargs):
        
        """Constructor"""
        

        if detectors:
            self.detectors = detectors
        else:
            self.detectors = {}

            self.default_kwargs = {'use_torch'       : False, 
                                   'device'          : 'cpu',
                                   'reference_time'  : 1370692818,
                                   'config_file_path': None}
            self.default_kwargs.update(kwargs)
            
            for name in names:
                self.detectors[name] = GWDetector(name, **self.default_kwargs)
                
        return 
    
    @property
    def names(self):
        return list(self.detectors.keys())
    
    @property
    def device(self):
        return self.default_kwargs['device']
    
    def set_new_device(self, new_device):
        self.default_kwargs['device'] = new_device
        #update the device for each detector
        for ifo in self.names:
            self.detectors[ifo].device = new_device
        return
    
    def set_reference_time(self, reference_time):
        """Set the reference time for the detectors in the network"""
        for ifo in self.names:
            self.detectors[ifo].reference_time = reference_time
        
    

    def project_wave(self, hp, hc, ra, dec, polarization, t_gps=None):
        """
        Project the plus and cross gw polarizations into the detectors frames

        Args:
        -----
            hp, hc: numpy.ndarrays or torch.tensors
                Plus and cross polarizations of the wave.

            ra: float
                Right ascension of the source in rad.

            dec: float
                Declination of the source in rad.

            polarization: float
                polarizationarization angle of the wave in rad.
                
            t_gps: float
                GPS time of the event
                
        Returns:
        --------
            h: dict of either numpy.ndarray or torch.tensor or pycbc/gwpy TimeSeries
                Final gravitational wave signal (detector stain) 
                for each of the detectors in the network.
        """
        
        h = dict()
        for name, detector in self.detectors.items():
            h[name] = detector.project_wave(hp, hc, ra, dec, polarization, t_gps)        
        return h
    
    def time_delay_from_earth_center(self, ra, dec, t_gps=None):
        """ 
        Returns the time delay(s) from Earth Center for each detector in the network

        Args:
        -----
            ra: float or numpy.ndarray or torch.tensor 
                Right ascension of the source in rad.

            dec: float or numpy.ndarray or torch.tensor 
                Declination of the source in rad.

            t_gps: float or or numpy.ndarray or torch.tensor 
                GPS time of the event
        
        """
        delays = dict()
        for name, detector in self.detectors.items():
            delays[name] = detector.time_delay_from_earth_center(ra, dec, t_gps)
        return delays
