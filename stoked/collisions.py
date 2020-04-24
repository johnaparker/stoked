import numpy as np
from stoked import interactions
from stoked.forces import pairwise_central_force

def collisions_sphere(radius, kn):
    def F(r):
        nonlocal radius

        if np.isscalar(radius):
            Nparticles = len(r)
            radius = np.full(Nparticles, radius, dtype=float)

        T1 = np.add.outer(radius, radius)

        overlap = T1 - r
        overlap[overlap<0] = 0
        return kn*overlap**1.5

    return pairwise_central_force(F)

class collisions_sphere_interface(interactions):
    """
    Collision between spheres and a planar surface using the force model F = kn*overlap^1.5
    """
    def __init__(self, radii, kn, zpos=0):
        """
        Arguments:
            radii   particle radii
            kn      force constant
            zpos    z-position of the plane (default: 0)
        """
        self.radii = radii
        self.kn = kn
        self.zpos = zpos

    def force(self):
        Nparticles = len(self.position)
        if np.isscalar(self.radii):
            rad = np.full(Nparticles, self.radii, dtype=float)
        else:
            rad = np.asarray(self.radii, dtype=float)

        F = np.zeros_like(self.position)

        dz = self.position[:,2] - self.zpos
        overlap = dz - rad
        idx = overlap < 0
        F[idx,2] = np.sign(dz[idx])*self.kn*np.sqrt(-overlap[idx])**3

        return F

    def torque(self):
        return np.zeros_like(self.position)
