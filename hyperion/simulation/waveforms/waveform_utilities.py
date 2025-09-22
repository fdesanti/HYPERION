"""Helper functions for gw simulations"""

from astropy import constants
from astropy import units as u
from bilby.gw.conversion import bilby_to_lalsimulation_spins


c     = constants.c.value     #[m/s]
G     = constants.G.value     #['m3/(kg s2)]
M_sun = constants.M_sun.value #[kg]
pc_to_m_conversion = (u.pc).to(u.m)   #conversion factor parsec --> meters


def pc_to_Msun(dist):
    """
    Converts distance to Solar Masses using geometrized G=c=1 units.
    
    Args:
        dist (float): distance value in parsec

    Returns:
        dist_m (float): distance in Solar Masses units
    """
    dist_m = dist * pc_to_m_conversion
    return dist_m*(c**2)/(G*M_sun)
    
def Msun_to_sec(mass):
    """
    Converts Solar Masses to time [s].
    
    Args:
        mass (float): mass value in Solar Masses units
    
    Returns:
        sec (float): Solar masses converted into seconds
    """
    return (mass*M_sun)*(G/c**3)


def spin_conversion(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2, M, q, phase=0, reference_frequency=5):
    """
    Converts spin parameters to lalsimulation spin parameters.

    Args:
        theta_jn (float): angle between total angular momentum and line of sight [rad]
        phi_jl   (float): azimuthal angle of the line of sight in the total angular momentum frame [rad]
        tilt_1   (float): tilt angle of the primary spin with respect to the orbital angular momentum [rad]
        tilt_2   (float): tilt angle of the secondary spin with respect to the orbital angular momentum [rad]
        phi_12   (float): azimuthal angle between the two spins in the plane orthogonal to the orbital angular momentum [rad] 
        a_1      (float): dimensionless spin magnitude of the primary
        a_2      (float): dimensionless spin magnitude of the secondary
        M        (float): total mass of the binary 
        q        (float): mass ratio m2/m1 with m2 <= m1
        phase               (float, optional): orbital phase at the reference frequency [rad]. Defaults to 0.
        reference_frequency (float, optional): reference frequency [Hz]. Defaults to 5.
    """

    #compute component masses
    mass_1 = M / (1 + q)
    mass_2 = M * q / (1 + q)

    inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = bilby_to_lalsimulation_spins(
                                                            theta_jn            = float(theta_jn),
                                                            phi_jl              = float(phi_jl),
                                                            tilt_1              = float(tilt_1),
                                                            tilt_2              = float(tilt_2),
                                                            phi_12              = float(phi_12),
                                                            a_1                 = float(a_1),
                                                            a_2                 = float(a_2),
                                                            mass_1              = float(mass_1),
                                                            mass_2              = float(mass_2),
                                                            phase               = float(phase),
                                                            reference_frequency = reference_frequency,
                                                            )
  
    return inclination, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z

