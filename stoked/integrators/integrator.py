from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.constants import k as kb
from stoked.hydrodynamics import particle_wall_self_mobility, grand_mobility_matrix
import quaternion

def fluctuation(friction, temperature, dt):
    """Given a friction matrix, return the fluctuation matrix"""
    if temperature == 0:
        return np.zeros_like(friction)
    else:
        rhs = 2*kb*temperature*friction/dt
        return np.linalg.cholesky(rhs)

def random_force_isotropic(alpha, temperature, dt):
    noise = np.random.normal(size=alpha.shape) 
    beta = np.sqrt(2*kb*temperature/(alpha*dt))
    return beta*noise

def random_force_anisotropic(alpha, temperature, dt):
    noise = np.random.normal(size=alpha.shape[:2]) 
    beta = np.zeros_like(alpha)
    for i in range(len(alpha)):
        beta[i] = np.linalg.inv(alpha[i]) @ fluctuation(alpha[i], temperature, dt)

    return np.einsum('Nij,Nj->Ni', beta, noise)

class integrator:
    __metaclass__ = ABCMeta

    def __init__(self, grand_mobility_interval=10):
        self.grand_mobility_interval = grand_mobility_interval
        self.total_steps = 0

    def initialize(self, solver):
        self.solver = solver
        self.dt = solver.dt
        self.hydrodynamics = solver.hydrodynamic_coupling
        self.isotropic = solver.drag.isotropic

    def solve_forces(self):
        F = self.solver._total_force(self.solver.time, self.solver.position, self.solver.orientation)
        return F

    def random_force(self):
        if self.isotropic:
            return random_force_isotropic(self.alpha_T, self.solver.temperature, self.dt)
        else:
            return random_force_anisotropic(self.alpha_T_rot, self.solver.temperature, self.dt)

    def random_torque(self):
        if self.isotropic:
            return random_force_isotropic(self.alpha_R, self.solver.temperature, self.dt)
        else:
            return random_force_anisotropic(self.alpha_R_rot, self.solver.temperature, self.dt)

    def solve_torques(self):
        T = self.solver._total_torque(self.solver.time, self.solver.position, self.solver.orientation)
        return T

    def perform_constraints(self):
        for constraint in self.solver.constraint:
            constraint(self.solver.position, self.solver.orientation)

    def pre_step(self):
        self.solver._update_interactions(self.solver.time, self.solver.position, self.solver.orientation)

        self._solve_mobility()
        if self.hydrodynamics and self.total_steps % self.grand_mobility_interval == 0:
            self._solve_grand_mobility()

        if not self.isotropic:
            self.solve_rotation_matrix()

    def post_step(self):
        if self.solver.rotating:
            self.solver.orientation = np.normalized(self.solver.orientation)

        self.solver.time += self.dt
        self.perform_constraints()
        self.total_steps += 1

    def solve_rotation_matrix(self):
        self.rot = quaternion.as_rotation_matrix(self.solver.orientation)
        self.alpha_T_rot = np.einsum('Nij,Nj,Nlj->Nil', self.rot, self.alpha_T, self.rot)
        self.alpha_R_rot = np.einsum('Nij,Nj,Nlj->Nil', self.rot, self.alpha_R, self.rot)

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

    def _solve_grand_mobility(self):
        drag_T = np.zeros([self.solver.Nparticles, 3, 3], dtype=float)
        drag_R = np.zeros([self.solver.Nparticles, 3, 3], dtype=float)
        np.einsum('Nii->Ni', drag_T)[...] = self.alpha_T
        np.einsum('Nii->Ni', drag_R)[...] = self.alpha_R

        self.grand_M = grand_mobility_matrix(self.solver.position, drag_T, drag_R, self.solver.drag.viscosity)
        self.grand_N = fluctuation(self.grand_M, self.solver.temperature, self.dt)

        # if self.drag.isotropic:
            # np.einsum('Nii->Ni', drag_T)[...] = self.alpha_T
            # np.einsum('Nii->Ni', drag_R)[...] = self.alpha_R
        # else:
            # rot = quaternion.as_rotation_matrix(orientation)
            # drag_T[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_T, rot)
            # drag_R[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_R, rot)

    @abstractmethod
    def bd_step(self):
        raise NotImplementedError('brownian dynamics step not implemented for this integrator')

    @abstractmethod
    def ld_step(self):
        raise NotImplementedError('langevin dynamics step not implemented for this integrator')

    @abstractmethod
    def hbd_step(self):
        raise NotImplementedError('hydrodynamic-brownian dynamics step not implemented for this integrator')

    @abstractmethod
    def hld_step(self):
        raise NotImplementedError('hydrodynamic-langevin dynamics step not implemented for this integrator')
