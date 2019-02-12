import numpy as np
from scipy.constants import k as kb

class brownian_dynamics:
    def __init__(self, position, Fext, drag, temperature, dt):
        """
        Arguments:
            position[N,3]    initial position of N particles
            Fext(t, r)       external force given time t, position r[N,3] and returns force F[N,3]
            drag             drag coefficient (F = -(drag)*(velocity))
            temperature      system temperature
            dt               time-step
        """
        self.position = np.asarray(position)
        self.Fext = Fext
        self.drag = drag
        self.temperature = temperature
        self.dt = dt
        self.time = 0

        self.alpha = 1/self.drag
        self.beta = np.sqrt(2*kb*self.temperature/(dt*self.drag))

        self.velocity = np.zeros_like(position)

    def step(self):
        """
        Time-step the positions by dt
        """
        F = self.Fext(self.time, self.position)
        noise = np.random.normal(size=self.position.shape) 
        v1 = (self.alpha*F + self.beta*noise)
        r_predict = self.position + self.dt*v1
        self.time += self.dt

        F_predict = self.Fext(self.time, r_predict)
        v2 = (self.alpha*F_predict + self.beta*noise)
        self.velocity = 0.5*(v1 + v2)
        self.position = self.position + 0.5*self.dt*(v1 + v2)
