from . import integrator
import numpy as np

class basic_integrator(integrator):
    def __init__(self):
        pass

    def bd_step(self):
        F = self.solve_forces()
        Fr = self.random_force()
        F += Fr

        v1 = self.alpha_T*F
        self.solver.velocity = v1
        self.solver.position += self.dt*v1

        if self.solver.rotating:
            T = self.solve_torques()
            w1 = self.alpha_R*T
            w1_q = np.array([np.quaternion(*omega) for omega in w1])
            self.solver.orientation = (1 + w1_q*self.dt/2)*self.solver.orientation

    def ld_step(self):
        F = self.solve_forces()
        Fr = self.random_force()
        F += Fr

        mass = self.solver.mass
        v0 = self.solver.velocity
        v1 = self.alpha_T*F + (v0 - self.alpha_T*F)*np.exp(-self.dt/mass/self.alpha_T)

        dr = self.alpha_T*F*self.dt - self.alpha_T*mass*(self.alpha_T*F - v0)*(1 - np.exp(-self.dt/mass/self.alpha_T))

        self.solver.velocity = v1
        self.solver.position += dr

        if self.solver.rotating:
            raise NotImplemented('ld step with rotation')

    # def _step_hydrodynamics(self):
        # self._update_interactions(self.time, self.position, self.orientation)

        # F = self._total_force(self.time, self.position, self.orientation)
        # noise_T = np.random.normal(size=self.position.shape) 

        # T = self._total_torque(self.time, self.position, self.orientation)
        # noise_R = np.random.normal(size=self.position.shape) 

        # v1, w1 = self._get_velocity_hydrodynamics(F, T, noise_T, noise_R, self.orientation)

        # r_predict = self.position + self.dt*v1
        # w1_q = np.array([np.quaternion(*omega) for omega in w1])
        # o_predict = (1 + w1_q*self.dt/2)*self.orientation


        # self.time += self.dt
        # self._perform_constraints(r_predict, o_predict)
        # self._update_interactions(self.time, r_predict, o_predict)

        # F_predict = self._total_force(self.time, r_predict, o_predict)
        # T_predict =  self._total_torque(self.time, r_predict, o_predict)

        # v2, w2 = self._get_velocity_hydrodynamics(F_predict, T_predict, noise_T, noise_R, o_predict)

        # w2_q = np.array([np.quaternion(*omega) for omega in w2])
        # w_q = (w1_q + w2_q)/2

        # self.velocity = 0.5*(v1 + v2)
        # self.position += self.dt*self.velocity
        # self.angular_velocity = 0.5*(w1 + w2)
        # self.orientation = (1 + w_q*self.dt/2)*self.orientation

        # self._perform_constraints(self.position, self.orientation)
        # self.orientation = np.normalized(self.orientation)

    # def _get_velocity(self, alpha, drive, noise, orientation):
        # """
        # FOR INTERNAL USE ONLY

        # obtain velocity (or angular velocity) given alpha parameter, drive force/torque, noise, and orientation
        # """
        # if self.isotropic:
            # beta = np.sqrt(2*kb*self.temperature*alpha/self.dt)
            # velocity = alpha*drive + beta*noise
        # else:
            # rot = quaternion.as_rotation_matrix(orientation)
            # alpha = np.einsum('Nij,Nj,Nlj->Nil', rot, alpha, rot)

            # beta = np.zeros_like(alpha)
            # for i in range(self.Nparticles):
                # beta[i] = fluctuation(alpha[i], self.temperature, self.dt)

            # velocity = np.einsum('Nij,Nj->Ni', alpha, drive) + np.einsum('Nij,Nj->Ni', beta, noise)

        # return velocity

    # def _get_velocity_hydrodynamics(self, F, T, noise_T, noise_R, orientation):
        # """
        # FOR INTERNAL USE ONLY
        # """
        # drag_T = np.zeros([self.Nparticles, 3, 3], dtype=float)
        # drag_R = np.zeros([self.Nparticles, 3, 3], dtype=float)
        # if self.drag.isotropic:
            # np.einsum('Nii->Ni', drag_T)[...] = self.alpha_T
            # np.einsum('Nii->Ni', drag_R)[...] = self.alpha_R
        # else:
            # rot = quaternion.as_rotation_matrix(orientation)
            # drag_T[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_T, rot)
            # drag_R[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_R, rot)

        # M = grand_mobility_matrix(self.position, drag_T, drag_R, self.drag.viscosity)

        # grand_F = np.hstack([F.flatten(), T.flatten()]) 
        # grand_noise = np.hstack([noise_T.flatten(), noise_R.flatten()]) 
        # N = fluctuation(M, self.temperature, self.dt)

        # grand_velocity = M @ grand_F + N @ grand_noise
        # velocity = grand_velocity[:3*self.Nparticles].reshape([self.Nparticles,3])
        # angular_velocity = grand_velocity[3*self.Nparticles:].reshape([self.Nparticles,3])

        # return velocity, angular_velocity

