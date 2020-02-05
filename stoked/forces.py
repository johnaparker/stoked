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
        with np.errstate(divide='ignore'):
            F_ij = self.force_func(r_ij)   # N x N x 3
        np.einsum('iix->ix', F_ij)[...] = 0
        F_i = np.sum(F_ij, axis=1)

        return F_i

    def torque(self):
        T_i = np.zeros_like(self.position)
        return T_i

    def __add__(self, other):
        def new_func(r):
            return self.force_func(r) + other.force_func(r)

        return pairwise_force(new_func)

class pairwise_central_force(interactions):
    def __init__(self, force_func):
        """
        Arguments:
            force_func     force function of the form F(r[dim]) -> [dim]
        """
        self.force_func = force_func

    def force(self):
        Nparticles = len(self.position)

        r_ij = self.position[:,np.newaxis] - self.position[np.newaxis]   # N x N x 3
        r = np.linalg.norm(r_ij, axis=-1)
        with np.errstate(divide='ignore'):
            F = self.force_func(r)
            r_inv = 1/r

        F_ij = np.einsum('ij,ijx,ij->ijx', F, r_ij, r_inv)
        np.einsum('iix->ix', F_ij)[...] = 0
        F_i = np.sum(F_ij, axis=1)

        return F_i

    def torque(self):
        T_i = np.zeros_like(self.position)
        return T_i

    def __add__(self, other):
        def new_func(r):
            return self.force_func(r) + other.force_func(r)

        return pairwise_central_force(new_func)

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
