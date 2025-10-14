import numpy as np
from abc import ABCMeta, abstractmethod

class inertia:
    """
    Abstract base class for drag coeffecients
    """
    __metaclass__ = ABCMeta

    def __init__(self, density, isotropic=False):
        """
        Arguments:
            density        mass density of the object
            isotropic      (bool) True if the moment of inertia is isotropic
        """
        self.density = density
        self.isotropic = isotropic

    @abstractmethod
    def _mass(self):
        raise NotImplementedError('mass has not been implemented for this type of particle')

    @abstractmethod
    def _moment(self):
        raise NotImplementedError('moment has not been implemented for this type of particle')

    @property
    def mass(self):
        """particle mass"""
        return self._mass()

    @property
    def moment(self):
        """particle moment of inertia"""
        return self._moment()


class inertia_sphere(inertia):
    """
    Inertia values for a sphere
    """

    def __init__(self, radius, density):
        """
        Arguments:
            radii[N,3]     ellipsoid radii
            density        mass density of the object
        """
        super().__init__(density, isotropic=True)
        self.radius = np.asarray(radius, dtype=float)

    def _mass(self):
        return 4/3*np.pi*self.radius**3*self.density

    def _moment(self):
        M = self.mass
        return 2/5*M*self.radius**2

class inertia_ellipsoid(inertia):
    """
    Inertia values for an ellipsoid
    """

    def __init__(self, radii, density):
        """
        Arguments:
            radii[N,3]     ellipsoid radii
            density        mass density of the object
        """
        super().__init__(density, isotropic=True)
        self.radii = np.atleast_2d(np.asarray(radii, dtype=float))

    def _mass(self):
        V = 4/3*np.pi*np.product(self.radii, axis=1)
        return V*self.density

    def _moment(self):
        M = self.mass
        Ix = 1/5*M*(self.radii[:,1]**2 + self.radii[:,2]**2)
        Iy = 1/5*M*(self.radii[:,0]**2 + self.radii[:,2]**2)
        Iz = 1/5*M*(self.radii[:,0]**2 + self.radii[:,1]**2)

        return np.array([Ix, Iy, Iz]).T
