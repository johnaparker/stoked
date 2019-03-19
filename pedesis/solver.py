import numpy as np
from scipy.constants import k as kb
from collections.abc import Iterable
import quaternion
from abc import ABCMeta, abstractmethod

class interactions:
    """
    Abstract base class for particle interactions
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def force(self):
        """
        Return the force[N,3] based on the current state
        """
        pass

    @abstractmethod
    def torque(self):
        """
        Return the torque[N,3] based on the current state
        """
        pass
    
    @abstractmethod
    def update(self):
        """
        Update the current state
        """
        pass

    def _update(self, time, position, orientation):
        self.time = time
        self.position = position
        self.orientation = orientation
        self.update()

class brownian_dynamics:
    """
    Perform a Brownian dynamics simulation, with optional external and internal internal interactions and rotational dynamics
    """
    def __init__(self, *, temperature, dt, position, drag, orientation=None, 
                 force=None, torque=None, interactions=None):
        """
        Arguments:
            temperature        system temperature
            dt                 time-step
            position[N,3]      initial position of N particles
            drag               drag coefficients (of base type pedesis.drag)
            orientation[N]     initial orientation of N particles (as quaternions)
            force(t, r, q)     external force function given time t, position r[N,3], orientation q[N] and returns force F[N,3] (can be a list of functions)
            torque(t, r, q)    external torque function given time t, position r[N,3], orientation q[N] and returns torque T[N,3] (can be a list of functions)
            interactions       particle interactions (can be a list)
        """
        self.temperature = temperature
        self.dt = dt
        self.time = 0
        self.position = np.atleast_2d(np.asarray(position, dtype=float))
        self.drag = drag

        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=np.quaternion)
        else:
            self.orientation = None

        if force is None:
            self.force = [lambda t, rvec, orient: np.zeros_like(rvec)]
        elif not isinstance(force, Iterable):
            self.force = [force]
        else:
            self.force = force

        if torque is None:
            self.torque = [lambda t, rvec, orient: np.zeros_like(rvec)]
        elif not isinstance(force, Iterable):
            self.torque = [torque]
        else:
            self.torque = torque

        if interactions is None:
            self.interactions = None
        elif not isinstance(interactions, Iterable):
            self.interactions = [interactions]
        else:
            self.interactions = interactions

        self.alpha_T = np.zeros_like(self.position)
        self.beta_T = np.zeros_like(self.position)

        drag_T = self.drag.drag_T
        if self.drag.isotropic and not np.isscalar(drag_T):
            drag_T = drag_T[:,np.newaxis]

        self.alpha_T[...] = 1/drag_T
        self.beta_T[...] = np.sqrt(2*kb*self.temperature/(dt*drag_T))
        self.velocity = np.zeros_like(self.position)

        if self.orientation is not None:
            self.alpha_R = np.zeros_like(self.position)
            self.beta_R = np.zeros_like(self.position)

            drag_R = self.drag.drag_R
            if self.drag.isotropic and not np.isscalar(drag_R):
                drag_R = drag_R[:,np.newaxis]

            self.alpha_R[...] = 1/drag_Rs
            self.beta_R[...] = np.sqrt(2*kb*self.temperature/(dt*drag_Rs))

            self.angular_velocity = np.zeros_like(self.position)

    def step(self):
        """
        Time-step the positions (and orientations) by dt
        """
        self._update_interactions(self.time, self.position, self.orientation)

        F = self._total_force(self.time, self.position, self.orientation)
        noise_T = np.random.normal(size=self.position.shape) 
        v1 = (self.alpha_T*F + self.beta_T*noise_T)
        r_predict = self.position + self.dt*v1

        if self.orientation is not None:
            T = self._total_torque(self.time, self.position, self.orientation)
            noise_R = np.random.normal(size=self.position.shape) 
            w1 = (self.alpha_R*T + self.beta_R*noise_R)
            o_predict = (1 + w1*self.dt/2)*self.orientation

        self.time += self.dt

        F_predict =  self._total_force(self.time, r_predict, None)
        v2 = (self.alpha_T*F_predict + self.beta_T*noise_T)
        self.velocity = 0.5*(v1 + v2)
        self.position = self.position + 0.5*self.dt*(v1 + v2)

        if self.orientation is not None:
            T_predict =  self._total_torque(self.time, r_predict, None)
            w2 = (self.alpha_R*T_predict + self.beta_R*noise_R)
            self.angular_velocity = 0.5*(w1 + w2)
            self.orientation = (1 + (w1 + w2)*self.dt/4)

            self.orientation = np.normalized(self.orientation)

    def _update_interactions(self, time, position, orientation):
        """
        FOR INTERNAL USE ONLY

        update the interactions given a new (or predicted) position and orientation
        """
        if self.interactions is not None:
            for I in self.interactions:
                I._update(time, position, orientation)

    def _total_force(self, time, position, orientation):
        """
        FOR INTERNAL USE ONLY

        return the total force at a given time, position, and orientation
        """
        F = sum((force(time, position, orientation) for force in self.force))
        if self.interactions is not None:
            for I in self.interactions:
                F += I.force()
        return F

    def _total_torque(self, time, position, orientation):
        """
        FOR INTERNAL USE ONLY

        return the total torque at a given time, position, and orientation
        """
        T = sum((torque(time, position, orientation) for torque in self.torque))
        if self.interactions is not None:
            for I in self.interactions:
                T += I.torque()
        return T
