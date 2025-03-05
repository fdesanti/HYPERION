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

def get_detectors_configs(det_conf_path=None):
    """
    Load the detectors configuration file
    """
    if not det_conf_path:
        det_conf_path = f"{CONF_DIR}/detectors.yml"
    with open(det_conf_path, 'r') as file:
        det_configs = yaml.safe_load(file)    
    return det_configs

detectors_configs = get_detectors_configs()
available_detectors = list(detectors_configs.keys())


class GWDetector:
    """
    Class for a ground based GW Detector

    Args:
        name             (str): Name of the detector. The list of available detectors is stored in the ``available_detectors`` variable.
        reference_time (float): Reference GPS time. (Default is 1370692818)
        use_torch       (bool): If True, the class will use torch tensors for computations. (Default is False)
        device           (str): Device to use for computations. It can be either "cpu" or "cuda". Used only when ``use_torch`` is True. (Default is "cpu")
        config_file_path (str): Path to a custom configuration file. If provided, the ``name`` argument will be ignored.
    """

    def __init__(self, name=None, reference_time=1370692818, use_torch=False, device='cpu', config_file_path=None):
        
        #check device assertion
        if device != 'cpu':
            try:
                assert use_torch == True, "Cannot use GPU without using torch"
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
            param t_gps (float, int, list, numpy.ndarray or torch.Tensor): GPS Time of arrival of the source signal.
        
        Returns:
            lst (float or numpy.ndarray or torch.Tensor): Local Sidereal Time(s) in rad. 
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
            xangle (float): The orientation of the detector's arms with respect to local geographical direction, in rad. It is measured counterclock-wise from East to the bisector of the interferometer arms
            ra     (float): Right ascension of the source in rad.
            dec    (float): Declination of the source in rad.
            t_gps (float, numpy.ndarray or torch.Tensor): GPS time at which compute the detector response function
         
         Returns:
            a, b (tuple of float or numpy.ndarray or torch.Tensors): relative amplitudes of hplus and hcross.
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
        Computes the Antenna Pattern functions for the plus and cross polarizations of the wave.
        (See: Phys. Rev. D 58, 063001)

        Args:
            ra (float): Right ascension of the source in rad.
            dec (float): Declination of the source in rad.
            polarization (float): Polarizationarization angle of the wave in rad.
            t_gps (float, int, list , numpy.ndarray or torch.Tensor): GPS Time of arrival of the source signal.

        Returns:
            fplus, fcross (tuple of float, numpy.ndarray or torch.Tensors): Antenna pattern response functions for the plus and cross polarizations respectively
        '''
        
        xangle = self.arms_orientation_angle    
        
        ampl11, ampl12 = self._ab_factors(xangle, ra, dec, t_gps)
        
        fplus  = self.xp.sin(self.angle_between_arms)*(ampl11*self.xp.cos(2*polarization) + ampl12*self.xp.sin(2*polarization))
        fcross = self.xp.sin(self.angle_between_arms)*(ampl12*self.xp.cos(2*polarization) - ampl11*self.xp.sin(2*polarization))

        return fplus, fcross

    def project_wave(self, hp, hc, ra, dec, polarization, t_gps=None):
        r"""
        Projects the plus and cross gw polarizations into the detector frame

        .. math::
            h = F_+ h_+ + F_{\times} h_{\times}

        Args:
            hp           (numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Plus polarizations of the wave.
            hc           (numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Cross polarizations of the wave.
            ra           (float, numpy.ndarray or torch.Tensor): Right ascension of the source in rad.
            dec          (float, numpy.ndarray or torch.Tensor): Declination of the source in rad.
            polarization (float, numpy.ndarray or torch.Tensor): polarizationarization angle of the wave in rad.
            t_gps        (float, numpy.ndarray or torch.Tensor): GPS time of the event. If None, the reference time of the detector is used. (Default: None)
                
        Returns:
            h (numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Projected gravitational wave signal (detector stain).
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
        """
        Returns the time delay from the Earth Center to the detector for
        a signal from a certain sky location at a given GPS time.

        Args:
            ra    (float, numpy.ndarray or torch.Tensor): Right ascension of the source in rad.
            dec   (float, numpy.ndarray or torch.Tensor): Declination of the source in rad.
            t_gps (float, numpy.ndarray or torch.Tensor): GPS time of the event. If None, the reference time of the detector is used. (Default: None)
        """

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
            other_location (list, numpy.ndarray, torch.Tensor, GWDetector): Earth Geocenter coordinates of the location.
            ra            (float, array, tensor): Right ascension of the source in rad.
            dec           (float, array, tensor): Declination of the source in rad.
            t_gps         (float, array, tensor): GPS time of the event. If None, the reference time of the detector is used. (Default: None)

        Returns:
            dt (float, numpy.ndarray, torch.Tensor): The arrival time difference between the detectors.
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


#==========================================
#---------- Detector Network --------------
#========================================== 
class GWDetectorNetwork():
    """
    Class for a network of gravitational wave detectors.
    It accepts a list of detector names (or a list of GWDetector instances).
    The class allows for the gw projection on the detector network.

    Args:
        names (list): List of detector names. (Default is ['H1', 'L1', 'V1'])
        detectors (dict): Dictionary of GWDetector instances. If provided, the ``names`` argument will be ignored.
        kwargs: Additional keyword arguments to be passed to each GWDetector instance.
    """

    def __init__(self, names=['H1', 'L1', 'V1'], detectors=None, **kwargs):
        
        #create detectors
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
    
    @property
    def names(self):
        return list(self.detectors.keys())
    
    @property
    def device(self):
        return self.default_kwargs['device']
    
    def set_new_device(self, new_device):
        """Set a new device for the detectors in the network"""
        self.default_kwargs['device'] = new_device
        for ifo in self.names:
            self.detectors[ifo].device = new_device
        return
    
    def set_reference_time(self, reference_time):
        """Set the reference time for the detectors in the network"""
        for ifo in self.names:
            self.detectors[ifo].reference_time = reference_time
        
    def project_wave(self, hp, hc, ra, dec, polarization, t_gps=None):
        """
        Project the plus and cross polarizations into the detectors frames

        Args:
            hp           (numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Plus polarizations of the wave.
            hc           (numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Cross polarizations of the wave.
            ra           (float, numpy.ndarray or torch.Tensor): Right ascension of the source in rad.
            dec          (float, numpy.ndarray or torch.Tensor): Declination of the source in rad.
            polarization (float, numpy.ndarray or torch.Tensor): Polarizationarization angle of the wave in rad.
            t_gps        (float, numpy.ndarray or torch.Tensor): GPS time of the event. If None, the reference time of the detector is used. (Default: None)
                
        Returns:
            h (dict of either numpy.ndarray, torch.Tensor or pycbc/gwpy TimeSeries): Final gravitational wave signal (detector stain) for each of the detectors in the network.
        """
        h = dict()
        for name, detector in self.detectors.items():
            h[name] = detector.project_wave(hp, hc, ra, dec, polarization, t_gps)        
        return h
    
    def time_delay_from_earth_center(self, ra, dec, t_gps=None):
        """ 
        Returns the time delay(s) from Earth Center for each detector in the network        
        """
        delays = dict()
        for name, detector in self.detectors.items():
            delays[name] = detector.time_delay_from_earth_center(ra, dec, t_gps)
        return delays
    
class EinsteinTelescope(GWDetectorNetwork):
    """Wrapper class for the (triangular) Einstein Telescope detector"""
    def __init__(self, **kwargs):
        super().__init__(names=['E1', 'E2', 'E3'], detectors=None, **kwargs)
