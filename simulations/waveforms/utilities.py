"""Helper functions for gw simulations"""
import astropy.constants as constants
from astropy import units as u

c     = constants.c.value     #[m/s]
G     = constants.G.value     #['m3/(kg s2)]
M_sun = constants.M_sun.value #[kg]
pc_to_m_conversion = (u.pc).to(u.m)   #conversion factor parsec --> meters

def pc_to_Msun(dist):
    """
    Converts distance to Solar Masses using geometrized G=c=1 units.
    
    Args:
    -----
        - dist (float): distance value in parsec
    Return:
    ------
        - dist_m (float): distance in Solar Masses units
    """
    dist_m = dist * pc_to_m_conversion
    return dist_m*(c**2)/(G*M_sun)
    
def Msun_to_sec(mass):
    """
    Converts Solar Masses to time [s].
    
    Args:
    -----
        - mass (float): mass value in Solar Masses units
    Return:
    -------
        - sec (float): Solar masses converted into seconds
    """
    return (mass*M_sun)*(G/c**3)

