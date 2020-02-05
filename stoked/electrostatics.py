import numpy as np
from scipy import constants
from stoked import interactions
from stoked.forces import pairwise_central_force

def electrostatics(charges):
    """
    Free-space point electrostatic interactions
    """
    ke = 1/(4*np.pi*constants.epsilon_0)

    def F(r):
        if np.isscalar(charges):
            Nparticles = len(r)
            Q = np.full(Nparticles, charges, dtype=float)
        else:
            Q = np.asarray(charges, dtype=float)

        return ke*np.outer(Q, Q)/r**2

    return pairwise_central_force(F)

def double_layer_sphere(radius, potential, temperature=300, debye=27.6e-9, eps_m=80.4, zp=1):
    """
    Electrostatic interactions in a fluid medium for spheres
    """
    radius = np.asarray(radius, dtype=float)
    potential = np.asarray(potential, dtype=float)

    def F(r):
        nonlocal radius, potential
        Nparticles = len(r)
        if not np.ndim(radius):
            radius = np.full(Nparticles, radius, dtype=float)

        if not np.ndim(potential):
            potential = np.full(Nparticles, potential, dtype=float)

        factor = 16*np.pi*constants.epsilon_0*eps_m/debye \
                 *(constants.k*temperature/(zp*constants.elementary_charge))**2
    
        T1 = np.add.outer(radius, radius)
        if temperature == 0:
            T2 = np.ones_like(potential)
        else:
            T2 = np.tanh(zp*constants.elementary_charge*potential/(4*constants.k*temperature))
        T2 = np.outer(T2, T2)

        Q = factor*T1*T2*np.exp(-(r - T1)/debye)
        return Q

    return pairwise_central_force(F)

class double_layer_sphere_interface(interactions):
    """
    Electrostatic interactions in a fluid medium near an interface
    """
    def __init__(self, radius, potential, potential_interface, debye=27.6e-9, eps_m=80.4, zp=1, zpos=0):
        self.radius = np.asarray(radius, dtype=float)
        self.potential = np.asarray(potential, dtype=float)
        self.potential_interface = potential_interface
        self.debye = debye
        self.eps_m = eps_m
        self.zp = zp
        self.zpos = zpos

    def force(self):
        Nparticles = len(self.position)
        if not np.ndim(self.radius):
            radius = np.full(Nparticles, self.radius, dtype=float)
        else:
            radius = self.radius
        if not np.ndim(self.potential):
            potential = np.full(Nparticles, self.potential, dtype=float)
        else:
            potential = self.potential

        factor = 16*np.pi*constants.epsilon_0*self.eps_m/self.debye \
                 *(constants.k*self.temperature/(self.zp*constants.elementary_charge))**2

        dz = self.position[:,2] - self.zpos

        T1 = np.tanh(self.zp*constants.elementary_charge*self.potential_interface/(4*constants.k*self.temperature))
        if self.temperature == 0:
            T2 = np.ones_like(potential)
        else:
            T2 = np.tanh(self.zp*constants.elementary_charge*potential/(4*constants.k*self.temperature))
        Q = np.sign(dz)*factor*radius*T1*T2*np.exp(-(np.abs(dz) - radius)/self.debye)

        F = np.zeros_like(self.position)
        F[:,2] = Q
        return F

    def torque(self):
        return np.zeros_like(self.position)
