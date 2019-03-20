import numpy as np
from scipy import constants
from pedesis import interactions

class electrostatics(interactions):
    """
    Free-space point electrostatic interactions
    """
    def __init__(self, charges):
        # self.charges = pedesis.array(charges)
        self.charges = charges

    def force(self):
        Nparticles = len(self.position)
        if np.isscalar(self.charges):
            Q = np.full(Nparticles, self.charges, dtype=float)
        else:
            Q = np.asarray(self.charges, dtype=float)

        ke = 1/(4*np.pi*constants.epsilon_0)

        r_ijx = self.position[:,np.newaxis,:] - self.position[np.newaxis,...]
        with np.errstate(divide='ignore'):
            F_ijx = ke*np.einsum('i,j,ij,ijx->ijx', Q, Q, 1/np.sum(np.abs(r_ijx)**3, axis=-1), r_ijx)
        np.einsum('iix->x', F_ijx)[...] = 0
        
        F = np.sum(F_ijx, axis=1)

        return F

    def torque(self):
        return np.zeros_like(self.position)


class double_layer_spheres(interactions):
    """
    Electrostatic interactions in a fluid medium for spheres
    """
    def __init__(self, radius, potential, debye=27.6e-9, eps_m=80.4, zp=1):
        self.radius = np.asarray(radius, dtype=float)
        self.potential = np.asarray(potential, dtype=float)
        self.debye = debye
        self.eps_m = eps_m
        self.zp = zp

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

        r_ijx = self.position[:,np.newaxis,:] - self.position[np.newaxis,...]
        r_ij = np.linalg.norm(r_ijx, axis=-1)

        T1 = np.add.outer(radius, radius)
        T2 = np.arctanh(self.zp*constants.elementary_charge*potential/(4*constants.k*self.temperature))
        Q = factor*T1*np.multiply.outer(T2, T2)*np.exp(-(r_ij - T1)/self.debye)

        with np.errstate(divide='ignore'):
            F_ijx = np.einsum('ij,ij,ijx->ijx', Q, 1/(r_ij + 1e-20), r_ijx)

        np.einsum('iix->x', F_ijx)[...] = 0
        F = np.sum(F_ijx, axis=1)

        return F

    def torque(self):
        return np.zeros_like(self.position)

class double_layer_interface(interactions):
    """
    Electrostatic interactions in a fluid medium near an interface
    """
    pass
