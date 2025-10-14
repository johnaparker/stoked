import numpy as np

class interface:
    """A no-slip interface"""
    def __init__(self, z=0):
        """
        Arguments:
            z   z-position of the interface
        """
        self.z = z

def levi_civita():
    """return the levi-civita symbol"""

    eijk = np.zeros((3, 3, 3), dtype=float)
    eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
    eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1
    return eijk

def particle_wall_self_mobility(position, interface, viscosity, radius):
    """
    Construct the particle wall self-mobility matrix for a single particle

    Arguments:
        position[3]       position of particle
        interface         interface object
        viscosity         dynamic viscosity µ of surrounding fluid
        radius            particle radius
    """
    M = np.zeros([2, 2, 3, 3], dtype=float)
    h = (position[2] - interface.z)/radius

    gamma_T = 6*np.pi*viscosity*radius
    gamma_R = 6*np.pi*viscosity*radius**3

    a = 1/(16*gamma_T)*(9/h - 2/h**3 + 1/h**5)
    b = 1/(8*gamma_T)*(9/h - 4/h**3 + 1/h**5)
    M[0,0] = np.diag([a,a,b])

    a = 15/(64*gamma_R)*(1/h**3)
    b = 3/(32*gamma_R)*(1/h**3)
    M[1,1] = np.diag([a,a,b])

    return M

def grand_mobility_matrix(position, drag_T, drag_R, viscosity):
    """
    Construct the grand mobility matrix for a given cluster

    Arguments:
        position[N,3]     position of N particles
        drag_T[N,3,3]     3 by 3 translational drag tensors of N particles
        drag_R[N,3,3]     3 by 3 rotational drag tensors of N particles
        viscosity         dynamic viscosity µ of surrounding fluid
    """

    Nparticles = len(position)
    M = np.zeros([2, 3*Nparticles, 2, 3*Nparticles], dtype=float)

    ### block-diagonal components
    for i in range(Nparticles):
        idx = np.s_[0,3*i:3*i+3,0,3*i:3*i+3]
        M[idx] = drag_T[i]

        idx = np.s_[1,3*i:3*i+3,1,3*i:3*i+3]
        M[idx] = drag_R[i]

    ### Off block-diagonal components
    factor = 1/(8*np.pi*viscosity)
    eps = levi_civita()
    for i in range(Nparticles):
        for j in range(i+1, Nparticles):
            r_ijx = position[i] - position[j]
            r_ij = np.linalg.norm(r_ijx)

            I = np.identity(3, dtype=float)
            T = np.outer(r_ijx, r_ijx)/r_ij**2
            K = np.einsum('ijk,k->ij', eps, r_ijx)/r_ij

            ### TT coupling
            idx = np.s_[0,3*i:3*i+3,0,3*j:3*j+3]
            M[idx] = factor/r_ij*(I + T)
            idx2 = np.s_[0,3*j:3*j+3,0,3*i:3*i+3]
            M[idx2] = M[idx]

            ### RR coupling
            idx = np.s_[1,3*i:3*i+3,1,3*j:3*j+3]
            M[idx] = factor/(2*r_ij**3)*(3*T - I)
            idx2 = np.s_[1,3*j:3*j+3,1,3*i:3*i+3]
            M[idx2] = M[idx]

            ### RT coupling
            idx = np.s_[1,3*i:3*i+3,0,3*j:3*j+3]
            M[idx] = -factor/r_ij**2*(K)
            idx2 = np.s_[1,3*j:3*j+3,0,3*i:3*i+3]
            M[idx2] = -M[idx]

            ### TR coupling
            idx3 = np.s_[0,3*i:3*i+3,1,3*j:3*j+3]
            M[idx3] = -M[idx]
            idx4 = np.s_[0,3*j:3*j+3,1,3*i:3*i+3]
            M[idx4] = -M[idx2]

    return M.reshape([6*Nparticles, 6*Nparticles])
