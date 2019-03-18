import numpy as np
from scipy import constants
from pedesis import interactions

class electrostatics(interactions):
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
        F_ijx = ke*np.einsum('i,j,ij,ijx->ijx', Q, Q, 1/np.sum(np.abs(r_ijx)**3 + 1e-20, axis=-1), r_ijx)
        np.einsum('iix->x', F_ijx)[...] = 0
        
        F = np.sum(F_ijx, axis=1)

        return F

    def torque(self):
        return np.zeros_like(self.position)
