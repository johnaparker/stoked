from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import k as kb

def fluctuation(friction, temperature, dt):
    """Given a friction matrix, return the fluctuation matrix"""
    if T == 0:
        return np.zeros_like(friction)
    else:
        rhs = 2*kb*temperature*friction/dt
        return np.linalg.cholesky(rhs)

def random_force(alpha, temperature, dt):
    noise = np.random.normal(size=alpha.shape) 
    beta = np.sqrt(2*kb*temperature/(alpha*dt))
    return beta*noise

class integrator:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def initialize(self, solver):
        self.solver = solver
        self.dt = solver.dt

    def solve_forces(self):
        F = self.solver._total_force(self.solver.time, self.solver.position, self.solver.orientation)
        F += random_force(self.alpha_T, self.solver.temperature, self.dt)
        return F

    def solve_torques(self):
        T = self._total_torque(self.solver.time, self.solver.position, self.solver.orientation)
        T += random_force(self.alpha_R, self.solver.temperature, self.dt)
        return T

    def perform_constraints(self):
        for constraint in self.solver.constraint:
            constraint(self.solver.position, self.solver.orientation)

    def pre_step(self):
        self.solver._update_interactions(self.solver.time, self.solver.position, self.solver.orientation)
        self._solve_mobility()

    def post_step(self):
        if self.solver.rotating:
            self.solver.orientation = np.normalized(self.solver.orientation)

        self.solver.time += self.dt
        self.perform_constraints()

    def _solve_mobility(self):
        if self.solver.interface is not None:
            M_T = np.copy(self.solver.alpha_T)
            M_R = np.copy(self.solver.alpha_R)
            for i in range(self.solver.Nparticles):
                radius = self.solver.drag.radius[i] if self.solver.drag.radius.ndim else self.solver.drag.radius
                M_self = particle_wall_self_mobility(self.solver.position[i], self.solver.interface, self.solver.drag.viscosity, radius)
                M_T[i] -= M_self[0,0].diagonal()
                M_R[i] -= M_self[1,1].diagonal()
                self.alpha_T, self.alpha_R = M_T, M_R
        else:
            self.alpha_T, self.alpha_R = self.solver.alpha_T, self.solver.alpha_R

    @abstractmethod
    def bd_step(self):
        raise NotImplementedError('bd_step not implemented for this integrator')

    @abstractmethod
    def ld_step(self):
        raise NotImplementedError('ld_step not implemented for this integrator')
