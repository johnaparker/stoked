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
        raise NotImplementedError('translational drag has not be implemented for this type of particle')

    @abstractmethod
    def _drag_R(self):
        raise NotImplementedError('rotational drag has not be implemented for this type of particle')

    @property
    def drag_T(self):
        """translational drag coeffecient"""
        return self._drag_T()

    @property
    def drag_R(self):
        """rotational drag coeffecient"""
        return self._drag_R()

class sphere_drag(drag):
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

class ellipsoid_drag(drag):
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
        self.radii = np.asarray(radii, dtype=float)

    def _drag_T(self):
        a, b, c = self.radii

        D = np.zeros(3, dtype=float)

        integrand = lambda t: 1/np.sqrt((1 + t)*((b/a)**2 + t)*((c/a)**2 + t))
        chi_0 = a**2*quad(integrand, 0, np.inf)[0]

        for i, comp in enumerate([a,b,c]):
            integrand = lambda t: 1/((1 + t)*np.sqrt(((a/comp)**2 + t)*((b/comp)**2 + t)*((c/comp)**2 + t)))
            alpha_0 = quad(integrand, 0, np.inf)[0]

            D[i] = 16*np.pi*viscosity*a*b*c/(chi_0 + alpha_0*comp**2)

        return D

    def _drag_R(self):
        a, b, c = self.radii

        D = np.zeros(3, dtype=float)
        alpha_0 = np.zeros(3, dtype=float)

        for i, comp in enumerate([a,b,c]):
            integrand = lambda t: 1/((1 + t)*np.sqrt(((a/comp)**2 + t)*((b/comp)**2 + t)*((c/comp)**2 + t)))
            alpha_0[i] = quad(integrand, 0, np.inf)[0]

        factor = 16*np.pi*viscosity*a*b*c 

        D[0] = factor/(3*(r[1]**2*alpha_0[1] + r[2]**2*alpha_0[2]))*(r[1]**2 + r[2]**2)
        D[1] = factor/(3*(r[0]**2*alpha_0[0] + r[2]**2*alpha_0[2]))*(r[0]**2 + r[2]**2)
        D[2] = factor/(3*(r[0]**2*alpha_0[0] + r[1]**2*alpha_0[1]))*(r[0]**2 + r[1]**2)

        return F
