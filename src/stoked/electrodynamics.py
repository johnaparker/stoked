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

def polarizability_sphere(radius, eps, wavelength, eps_b=1, dynamic_correction=True):
    """
    Polarizability of a sphere. Returns (static alpha, dynamic alpha)

    Arguments:
        radius         sphere radius
        eps            material permitivitty
        wavelength     wavelength of light
        eps_b          background permitivitty
        dynamic_correction    whether to dynamically correct the static polarizability
    """
    radius = np.atleast_1d(radius)
    alpha_0 = 4*np.pi*eps_b*constants.epsilon_0*radius**3 * (eps - eps_b)/(eps + 2*eps_b)
    k = 2*np.pi*eps_b**0.5/wavelength

    correction = 1
    if dynamic_correction:
        correction -= 1j*k**3*alpha_0/(6*np.pi*eps_b*constants.epsilon_0)
        correction -= k**2*alpha_0/(6*np.pi*eps_b*constants.epsilon_0*radius)

    alpha = alpha_0/correction

    return alpha_0, alpha

class point_dipole_electrodynamics(interactions):
    """Perform point dipole electrodynamic interactions"""
    def __init__(self, alpha, source, wavelength, eps_b=1, interactions=True):
        """Arguments:
                alpha           (static, dynamic) polarizability of the particles
                source          incident source object that defines E and H functions
                wavelength      wavelength of illumination (in meters)
                eps_b           background permitivitty (default: 1)
                interactions    include particle interactions (default: True)
                use_dynamic_polarizability       modify the input alpha to be frequency corrected
        """

        self.Nparticles = len(alpha[0])

        self.alpha_0 = np.zeros([self.Nparticles,3,3], dtype=complex)
        self.alpha = np.zeros([self.Nparticles,3,3], dtype=complex)

        np.einsum('Nii->Ni', self.alpha_0)[...] = alpha[0]
        np.einsum('Nii->Ni', self.alpha)[...] = alpha[1]

        self.alpha_0_t = np.copy(self.alpha_0)
        self.alpha_t = np.copy(self.alpha)

        self.source = source
        self.wavelength = wavelength
        self.eps_b = eps_b
        self.k = 2*np.pi*self.eps_b**0.5/self.wavelength
        self.interactions = interactions

        self.sol = None
        self.G = np.zeros(shape = (self.Nparticles, 3, self.Nparticles, 3), dtype=complex)

    def solve(self):
        """Solve for the electric fields given the current state of particles"""

        Einc = self.source.E_field(*self.position.T, self.k).T

        rot = quaternion.as_rotation_matrix(self.orientation)
        self.alpha_t[...] = np.einsum('Nij,Njk,Nlk->Nil', rot, self.alpha, rot)
        self.alpha_0_t[...] = np.einsum('Nij,Njk,Nlk->Nil', rot, self.alpha_0, rot)

        if not self.interactions:
            self.sol = Einc
        else:
            identity = np.zeros(shape = (self.Nparticles, 3, self.Nparticles, 3), dtype=complex)
            np.einsum('ixix->xi', identity)[...] = 1

            for i in range(self.Nparticles):
                for j in range(self.Nparticles):
                    if i == j: continue
                    pi = self.position[i]
                    pj = self.position[j]
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
            E = self.source.E_field(*r.T, self.k).T
        else:
            E = np.zeros_like(r, dtype=np.complex)

        for i in range(self.Nparticles):
            R = r - self.particles.position[i]
            G = greens_any_dim(R, self.k, self.eps_b, delta)
            E += np.einsum('...xy,yz,z', G, self.alpha_t[i], self.sol[i]) 
        return E

    def update(self):
        self.solve()

    def force(self):
        """Return the force and torque on all particles, (F[N,3],T[N,3])"""

        def Efield_without(r, i):
            """Compute Efield without particle i"""
            E = self.source.E_field(*r, self.k)
            for j in range(self.Nparticles):
                if i == j: continue
                R = r - self.position[j]
                G = greens(R, self.k, self.eps_b)
                E += np.einsum('xy,yz,z', G, self.alpha_t[j], self.sol[j]) 
            return E

        F = np.zeros([self.Nparticles, 3])

        eps = 1e-12
        for i in range(self.Nparticles):
            Edx = Efield_without(self.position[i] + eps*np.array([1,0,0]), i)
            Edy = Efield_without(self.position[i] + eps*np.array([0,1,0]), i)
            Edz = Efield_without(self.position[i] + eps*np.array([0,0,1]), i)
            dE = (np.array([Edx,Edy,Edz]) - self.sol[i])/eps

            F[i] = 0.5*np.real(
                np.einsum('xy,y,zx', np.conj(self.alpha_t[i]), np.conj(self.sol[i]), dE))

        return F

    def torque(self):
        """Return the force and torque on all particles, (F[N,3],T[N,3])"""
        T = np.zeros([self.Nparticles, 3])

        for i in range(self.Nparticles):
            p = np.dot(self.alpha_t[i],self.sol[i])
            a0_inverse = np.linalg.inv(self.alpha_0_t[i])
            T[i] = 0.5*np.real(
                np.cross(np.conj(p), np.dot(a0_inverse, p)))

        return T
