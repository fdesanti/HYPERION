import os
import glob
import json

from astropy import units as u
from astropy.time import Time
from astropy.constants import c
from astropy.coordinates import EarthLocation
c = c.value #speed of light value

from ..config import CONF_DIR


def list_available_detectors():
    #list all the detector files in conf dir
    full_path = glob.glob(f"{CONF_DIR}/detectors/*_detector.json")
    #extract the detector names
    available = sorted([os.path.basename(path)[:-14] for path in full_path])
    return available

available_detectors = list_available_detectors()


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
                print('[WARNING]: ', e)
                print('[INFO]: Setting use_torch to True')
                use_torch = True
        
        self.device    = device
        
        if use_torch:
            self.xp = __import__('torch')
        else:
            self.xp = __import__('numpy')
        self.use_torch = use_torch
        
                  
        if name is not None:
            assert name in available_detectors, f"{name} detector not available. Available ones are {available_detectors}. Please select one of those or provide a custom config file path"
            config_file_path = f"{CONF_DIR}/detectors/{name}_detector.json"
        else:
            assert config_file_path is not None, "No name specified, please provide at least a custom detector config file"
            
            
        #loading configuration parameters
        with open(config_file_path) as config_file:
            det_conf = json.load(config_file)
            conf_params = det_conf['config_parameters']
            
            self.name                   = det_conf['name']
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
        return self.lst_estimate(t_gps, sidereal_time_kwargs={'kind':'mean', 'longitude': 'greenwich'})
          
          
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
            hp, hc: either numpy.ndarray or torch.tensor or pycbc/gwpy TimeSeries
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
            h: either numpy.ndarray or torch.tensor or pycbc/gwpy TimeSeries
                Final gravitational wave signal (detector stain).
        """
        #assert(len(hp)==len(hc))
        #t_gpss = hp.get_sampled_time()
        if t_gps is None:
            t_gps = self.reference_time
        f_plus, f_cross = self.antenna_pattern_functions(ra, dec, polarization, t_gps)

        h_signal = hp*f_plus + hc*f_cross
        
        return h_signal
    
    def time_delay_from_earth_center(self, ra, dec, t_gps=None):
        """Returns the time delay from Earth Center"""
        if t_gps is None:
            t_gps = self.reference_time
        return self.time_delay_from_location([0, 0, 0], ra, dec, t_gps)
    
    
    def time_delay_from_location(self, other_location, ra, dec, t_gps = None):
        """---- Adapted from PyCBC ----"""
        
        """Return the time delay from the given location to detector for
        a signal with the given sky location
        In other words return `t1 - t2` where `t1` is the
        arrival time in this detector and `t2` is the arrival time in the
        other location.

        Parameters:
        -----------
        other_location : list of Earth Geocenter coordinates or GWDetector instance
            
        ra : float
            The right ascension (in rad) of the signal.
        declination : float
            The declination (in rad) of the signal.
        t_gps : float
            The GPS time (in s) of the signal.

        Returns
        -------
        float
            The arrival time difference between the detectors.
        """

        if t_gps is None:
            t_gps = self.reference_time
        
        if isinstance(other_location, GWDetector):
            other_location = other_location.location
        
        ra_angle = self.gmst(t_gps) - ra
        cosd     = self.xp.cos(dec)

        e0 = cosd * self.xp.cos(ra_angle)
        e1 = cosd * -self.xp.sin(ra_angle)
        e2 = self.xp.sin(dec)

        if e0.ndim > 0:
            ehat = self.xp.concatenate([e0, e1, e2], -1)
        else:
            ehat = [e0, e1, e2]
            ehat = self.xp.tensor(ehat) if self.use_torch else self.xp.array(ehat)
        
        other_location = self.xp.tensor(other_location, dtype=self.xp.float64).to(self.device) if self.use_torch else self.xp.array(other_location)
        dx = other_location - self.location
        #print((dx*ehat).shape)
        #print(((dx * ehat).sum(axis = -1, keepdim=True) / c).shape)
        kwargs = {'keepdim':True} if self.use_torch else {'keepdims':True}
        return (dx * ehat).sum(axis = -1, **kwargs) / c



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

    def __init__(self, detectors=None, names=['H1', 'L1', 'V1'], **kwargs):
        
        """Constructor"""
        

        if detectors:
            self.detectors = detectors
        else:
            self.detectors = {}

            default_kwargs = {'use_torch': False, 
                              'device': 'cpu', 
                              'reference_time':1370692818, 
                              'config_file_path' : None}
            default_kwargs.update(kwargs)
            
            for name in names:
                self.detectors[name] = GWDetector(name, **default_kwargs)

        return 
    

    @property
    def names(self):
        return list(self.detectors.keys())
    

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