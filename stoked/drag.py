import numpy as np
from abc import ABCMeta, abstractmethod
from scipy.integrate import quad

class drag:
    """
    Abstract base class for drag coeffecients
    """
    __metaclass__ = ABCMeta

    def __init__(self, viscosity, isotropic=False):
        """
        Arguments:
            viscosity      dynamic viscosity µ of surrounding fluid
            isotropic      (bool) True if the drag is isotropic
        """
        self.viscosity = viscosity
        self.isotropic = isotropic

    @abstractmethod
    def _drag_T(self):
        raise NotImplementedError('translational drag has not beem implemented for this type of particle')

    @abstractmethod
    def _drag_R(self):
        raise NotImplementedError('rotational drag has not been implemented for this type of particle')

    @property
    def drag_T(self):
        """translational drag coeffecient"""
        return self._drag_T()

    @property
    def drag_R(self):
        """rotational drag coeffecient"""
        return self._drag_R()

class drag_sphere(drag):
    """
    Drag coeffecients for a sphere
    """

    def __init__(self, radius, viscosity):
        """
        Arguments:
            radius[N]      sphere radii
            viscosity      dynamic viscosity µ of surrounding fluid
        """
        super().__init__(viscosity, isotropic=True)
        self.radius = np.asarray(radius, dtype=float)

    def _drag_T(self):
        return 6*np.pi*self.radius*self.viscosity

    def _drag_R(self):
        return 8*np.pi*self.radius**3*self.viscosity

class drag_ellipsoid(drag):
    """
    Drag coeffecients for an ellipsoid
    """

    def __init__(self, radii, viscosity):
        """
        Arguments:
            radii[N,3]     ellipsoid radii
            viscosity      dynamic viscosity µ of surrounding fluid
        """
        super().__init__(viscosity)
        self.radii = np.atleast_2d(np.asarray(radii, dtype=float))
        self.chi_0 = np.zeros(len(self.radii), dtype=float)
        self.alpha_0 = np.zeros_like(self.radii)

        for i in range(len(self.radii)):
            a, b, c = self.radii[i]
            integrand = lambda t: 1/np.sqrt((1 + t)*((b/a)**2 + t)*((c/a)**2 + t))
            self.chi_0[i] = b*c*quad(integrand, 0, np.inf)[0]

            for j, comp in enumerate([a,b,c]):
                integrand = lambda t: 1/((1 + t)*np.sqrt(((a/comp)**2 + t)*((b/comp)**2 + t)*((c/comp)**2 + t)))
                self.alpha_0[i][j] = a*b*c/comp*quad(integrand, 0, np.inf)[0]

    def _drag_T(self):
        D = np.zeros_like(self.radii)
        for i in range(len(self.radii)):
            a, b, c = self.radii[i]
            for j, comp in enumerate([a,b,c]):
                D[i][j] = 16*np.pi*self.viscosity*a*b*c/(self.chi_0[i] + self.alpha_0[i][j])

        return D

    def _drag_R(self):
        D = np.zeros_like(self.radii)
        for i in range(len(self.radii)):
            a, b, c = self.radii[i]
            factor = 16*np.pi*self.viscosity*a*b*c 

            D[i][0] = factor/(3*(self.alpha_0[i][1] + self.alpha_0[i][2]))*(b**2 + c**2)
            D[i][1] = factor/(3*(self.alpha_0[i][0] + self.alpha_0[i][2]))*(a**2 + c**2)
            D[i][2] = factor/(3*(self.alpha_0[i][0] + self.alpha_0[i][1]))*(a**2 + b**2)

        return D
