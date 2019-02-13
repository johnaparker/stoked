import numpy as np
from scipy import constants

def electrostatics(charges):
    def force(t, rvec):
        Nparticles = len(rvec)

        if np.isscalar(charges):
            Q = np.full(Nparticles, charges, dtype=float)
        else:
            Q = charges

        ke = 1/(4*np.pi*constants.epsilon_0)

        r_ijx = rvec[:,np.newaxis,:] - rvec[np.newaxis,...]
        F_ijx = ke*np.einsum('i,j,ij,ijx->ijx', Q, Q, 1/np.sum(np.abs(r_ijx)**3 + 1e-20, axis=-1), r_ijx)
        np.einsum('iix->x', F_ijx)[...] = 0
        
        F = np.sum(F_ijx, axis=1)

        return F

    return force
