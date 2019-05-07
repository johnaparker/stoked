"""
Electrodynamic interactions in the point-dipole approximation
"""
import numpy as np
from scipy import constants
import quaternion
from stoked import interactions


def greens(R, k, eps_b=1):
    """Return the Green's matrix"""
    r = np.linalg.norm(R)
    A = np.exp(1j*k*r)/(4*np.pi*constants.epsilon_0*eps_b*r**3)
    B = (3 - 3j*k*r - k**2*r**2)/r**2 * np.outer(R,R)
    C = (k**2*r**2 + 1j*k*r - 1) * np.identity(3)
    return A*(B + C)

def greens_any_dim(R, k, eps_b=1, delta=0):
    """Return the Green's matrix"""
    # R shape: [...,3]
    r = np.linalg.norm(R, axis=-1) + delta # shape [...]
    A = np.exp(1j*k*r)/(4*np.pi*constants.epsilon_0*eps_b*r**3)  # shape [...]
    B = np.einsum('...,...i,...j->...ij', (3 - 3j*k*r - k**2*r**2)/r**2, R, R)  # shape [...,3,3]
    C = np.einsum('...,ij->...ij', k**2*r**2 + 1j*k*r - 1, np.identity(3))  # shape [...,3,3]
    return np.einsum('...,...ij->...ij', A, B + C)
    #return np.einsum(A, [Ellipsis], B+C, [Ellipsis, 0, 1])

class point_dipole_electrodynamics(interactions):
    """Perform point dipole electrodynamic interactions"""
    def __init__(self, alpha_0, E_field, wavelength, eps_b=1, interactions=True, use_dynamic_polarizability=True):
        """Arguments:
                particles       particle system
                alpha_0[N,3]    static polarizability vectors of N particles (alternative shape: [3], CONST)
                source          incident source object that defines E and H functions
                wavelength      wavelength of illumination (in meters)
                eps_b           background permitivitty (default: 1)
                interactions    include particle interactions (default: True)
                use_dynamic_polarizability       modify the input alpha to be frequency corrected
        """

        self.particles = particles
        self.N = particles.Nparticles
        self.alpha_0 = np.zeros([self.N,3,3], dtype=complex)
        np.einsum('Nii->Ni', self.alpha_0)[...] = alpha_0

        self.source = source
        self.wavelength = wavelength
        self.eps_b = eps_b
        self.k = 2*np.pi*self.eps_b**0.5/self.wavelength
        self.interactions = interactions

        self.sol = None
        self.G = np.zeros(shape = (self.N, 3, self.N, 3), dtype=np.complex)

        particles.interactions.append(self)

        if use_dynamic_polarizability:
            # self.alpha = self.alpha_0/(1 - 2/3*1j*self.k**3*self.alpha_0)
            self.alpha = self.alpha_0/(1 - 1j*self.k**3*self.alpha_0/(6*np.pi*eps_b*constants.epsilon_0))
        else:
            self.alpha = self.alpha_0

        self.alpha_0_t = np.copy(self.alpha_0)
        self.alpha_t = np.copy(self.alpha)

    def solve(self):
        """Solve for the electric fields given the current state of particles"""

        Einc = self.source.E(self.particles.position.T, self.k).T

        rot = quaternion.as_rotation_matrix(self.particles.orientation)
        self.alpha_t[...] = np.einsum('Nij,Njk,Nlk->Nil', rot, self.alpha, rot)
        self.alpha_0_t[...] = np.einsum('Nij,Njk,Nlk->Nil', rot, self.alpha_0, rot)

        if not self.interactions:
            self.sol = Einc
        else:
            identity = np.zeros(shape = (self.N, 3, self.N, 3), dtype=np.complex)
            np.einsum('ixix->xi', identity)[...] = 1

            for i in range(self.N):
                for j in range(self.N):
                    if i == j: continue
                    pi = self.particles.position[i]
                    pj = self.particles.position[j]
                    dji = pi -  pj
                    self.G[i,:,j,:] = greens(dji, self.k, self.eps_b)
                    # G[j,:,i,:] = G[i,:,j,:]

            A = identity - np.einsum('ixjy,jyz->ixjz', self.G, self.alpha_t)
            self.sol = np.linalg.tensorsolve(A, Einc)

    def Efield(self, r, inc=True, delta=1e-13):
        """Compute the electic field at positions r[...,3] due to all particles

           Arguments:
                 r         positions (shape: [...,3])
                 inc       include the incident field (default: True)
                 delta     pad the distance between point and scatterers by delta (to avoid singularities)
        """

        if inc:
            E = self.source.E(r.T, self.k).T
        else:
            E = np.zeros_like(r, dtype=np.complex)

        for i in range(self.N):
            R = r - self.particles.position[i]
            G = greens_any_dim(R, self.k, self.eps_b, delta)
            E += np.einsum('...xy,yz,z', G, self.alpha_t[i], self.sol[i]) 
        return E

    def force_and_torque(self):
        """Return the force and torque on all particles, (F[N,3],T[N,3])"""
        self.solve()

        def Efield_without(r, i):
            """Compute Efield without particle i"""
            E = self.source.E(r, self.k)
            for j in range(self.N):
                if i == j: continue
                R = r - self.particles.position[j]
                G = greens(R, self.k, self.eps_b)
                E += np.einsum('xy,yz,z', G, self.alpha_t[j], self.sol[j]) 
            return E

        F = np.zeros([self.N, 3])
        T = np.zeros([self.N, 3])

        eps = 1e-12
        for i in range(self.N):
            Edx = Efield_without(self.particles.position[i] + eps*np.array([1,0,0]), i)
            Edy = Efield_without(self.particles.position[i] + eps*np.array([0,1,0]), i)
            Edz = Efield_without(self.particles.position[i] + eps*np.array([0,0,1]), i)
            dE = (np.array([Edx,Edy,Edz]) - self.sol[i])/eps

            F[i] = 0.5*np.real(
                np.einsum('xy,y,zx', np.conj(self.alpha_t[i]), np.conj(self.sol[i]), dE))

            p = np.dot(self.alpha_t[i],self.sol[i])
            a0_inverse = np.linalg.inv(self.alpha_0_t[i])
            T[i] = 0.5*np.real(
                np.cross(np.conj(p), np.dot(a0_inverse, p)))

        return F,T
