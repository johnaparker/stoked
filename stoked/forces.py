import numpy as np
from stoked import interactions

class pairwise_force(interactions):
    """
    Pair-wise interaction force
    """
    def __init__(self, force_func):
        """
        Arguments:
            force_func     force function of the form F(r[dim]) -> [dim]
        """
        self.force_func = force_func

    def force(self):
        Nparticles = len(self.position)

        r_ij = self.position[:,np.newaxis] - self.position[np.newaxis]   # N x N x 3
        F_ij = self.force_func(r_ij)   # N x N x 3
        np.einsum('iix->ix', F_ij)[...] = 0
        F_i = np.sum(F_ij, axis=1)

        return F_i

    def torque(self):
        T_i = np.zeros_like(self.position)
        return T_i

def pairwise_central_force(force_func):
    def F(rvec):
        r = np.linalg.norm(rvec, axis=-1)
        f = force_func(r)
        r_inv = 1/r
        np.einsum('ii->i', r_inv)[...] = 0

        return np.einsum('ij,ijx,ij->ijx', f, rvec, r_inv)

    return pairwise_force(F)

def pairwise_potential(potential_func):
    pass
    # def F(rvec):
        # eps = 1e-15*np.ones_like(rvec)
        # U1 = potential_func(rvec)
        # U2 = potential_func(rvec + eps)
         # = -(U2 - U1)/eps


    # return pairwise_force(F)

def pairwise_central_potential():
    pass
