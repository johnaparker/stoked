import numpy as np
from stoked import interactions

class van_der_waals_sphere(interactions):
    def __init__(self, radius, hamaker):
        self.radius = np.asarray(radius, dtype=float)
        self.hamaker = hamaker

    def force(self):
        Nparticles = len(self.position)
        if not np.ndim(self.radius):
            radius = np.full(Nparticles, self.radius, dtype=float)
        else:
            radius = self.radius

        factor = 2*self.hamaker/3
        r_ijx = self.position[:,np.newaxis,:] - self.position[np.newaxis,...]
        r_ij = np.linalg.norm(r_ijx, axis=-1)

        T1 = np.multiply.outer(radius, radius)
        T2 = np.add.outer(radius, radius)**2
        T3 = np.substract.outer(radius, radius)**2
        Q = r_ij*T1*(1/(r_ij**2 - T2**2) - 1 /(r_ij**2 - T3**2))**2

        F_ijx = np.einsum('ij,ij,ijx->ijx', -factor*Q, 1/(r_ij + 1e-20), r_ijx)
        np.einsum('iix->x', F_ijx)[...] = 0
        F = np.sum(F_ijx, axis=1)

        return F

    def torque(self):
        return np.zeros_like(self.position)


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
        return F

    def torque(self):
        return np.zeros_like(self.position)
