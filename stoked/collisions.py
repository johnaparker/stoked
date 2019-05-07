import numpy as np
from stoked import interactions

class collisions_sphere(interactions):
    """
    Collision between spheres using the force model F = kn*overlap^1.5
    """
    def __init__(self, radii, kn):
        """
        Arguments:
            radii   particle radii
            kn      force constant
        """
        # self.radii = stoked.array(radii)
        self.radii = radii
        self.kn = kn

    def force(self):
        Nparticles = len(self.position)
        if np.isscalar(self.radii):
            rad = np.full(Nparticles, self.radii, dtype=float)
        else:
            rad = np.asarray(self.radii, dtype=float)

        F = np.zeros_like(self.position)

        for i in range(0,Nparticles):
            for j in range(i+1,Nparticles):
                d = self.position[i] - self.position[j]
                r = np.linalg.norm(d)
                if r < rad[i] + rad[j]:
                    overlap = abs(r - rad[i] + rad[j])

                    r_hat = d/r
                    Fn = r_hat*self.kn*overlap**1.5

                    F[i] += Fn
                    F[j] -= Fn
        return F

    def torque(self):
        return np.zeros_like(self.position)


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
        idx = np.abs(dz) < rad
        F[idx,2] = np.sign(dz[idx])*self.kn*np.sqrt((dz[idx] - rad[idx])**3)

        return F

    def torque(self):
        return np.zeros_like(self.position)
