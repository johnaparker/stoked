import numpy as np
from scipy.constants import k as kb

class brownian_dynamics:
    def __init__(self, position, drag, temperature, dt, force=None, torque=None):
        """
        Arguments:
            position[N,3]    initial position of N particles
            drag             drag coefficient (F = -(drag)*(velocity))
            temperature      system temperature
            dt               time-step
            force(t, r)      external force function given time t, position r[N,3] and returns force F[N,3]
            torque(t, r)      external torque function given time t, position r[N,3] and returns torque T[N,3]
        """
        self.position = np.asarray(position)
        self.drag = drag
        self.temperature = temperature
        self.dt = dt
        self.time = 0

        if force is None:
            self.force = lambda t, rvec: np.zeros_like(rvec)
        else:
            self.force = force

        if torque is None:
            self.torque = lambda t, rvec: np.zeros_like(rvec)
        else:
            self.torque = torque

        self.alpha = np.zeros_like(self.position)
        self.beta = np.zeros_like(self.position)

        if drag.ndim == 1:
            drag_s = self.drag[:,np.newaxis]
        else:
            drag_s = self.drag

        self.alpha[...] = 1/drag_s
        self.beta[...] = np.sqrt(2*kb*self.temperature/(dt*drag_s))

        self.velocity = np.zeros_like(position)

    def step(self):
        """
        Time-step the positions by dt
        """
        F = self.force(self.time, self.position)
        noise = np.random.normal(size=self.position.shape) 
        v1 = (self.alpha*F + self.beta*noise)
        r_predict = self.position + self.dt*v1
        self.time += self.dt

        F_predict = self.force(self.time, r_predict)
        v2 = (self.alpha*F_predict + self.beta*noise)
        self.velocity = 0.5*(v1 + v2)
        self.position = self.position + 0.5*self.dt*(v1 + v2)
