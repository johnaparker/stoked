import numpy as np
from stoked import interactions
from stoked.forces import pairwise_central_force

def van_der_waals_sphere(radius, hamaker):
    radius = np.asarray(radius, dtype=float)

    def F(r):
        nonlocal radius

        Nparticles = len(r)
        if not np.ndim(radius):
            radius = np.full(Nparticles, radius, dtype=float)

        factor = 2*self.hamaker/3
        T1 = np.outer(radius, radius)
        T2 = np.add.outer(radius, radius)**2
        T3 = np.substract.outer(radius, radius)**2
        Q = r*T1*(1/(r**2 - T2**2) - 1 /(r**2 - T3**2))**2

        return Q

    return F

class van_der_waals_sphere_interface(interactions):
    def __init__(self, radius, hamaker, zpos=0):
        self.radius = np.asarray(radius, dtype=float)
        self.hamaker = hamaker
        self.zpos = zpos

    def force(self):
        Nparticles = len(self.position)
        if not np.ndim(self.radius):
            radius = np.full(Nparticles, self.radius, dtype=float)
        else:
            radius = self.radius

        factor = self.hamaker/6
        dz = self.position[:,2] - self.zpos

        Q = radius*(1/(dz - radius) - 1/(dz + radius))**2

        F = np.zeros_like(self.position)
        F[:,2] = -np.sign(dz)*factor*Q
        idx = (dz - radius < 3e-9)
        F[idx] = 0

        return F

    def torque(self):
        return np.zeros_like(self.position)
