from . import integrator
import numpy as np

class basic_integrator(integrator):
    def __init__(self, grand_mobility_interval=10):
        super().__init__(grand_mobility_interval)

    def bd_step(self):
        F = self.solve_forces()
        F += self.random_force()

        if self.isotropic:
            v1 = self.alpha_T*F
        else:
            v1 = np.einsum('Nij,Nj->Ni', self.alpha_T_rot, F)

        self.solver.velocity = v1
        self.solver.position += self.dt*v1

        if self.solver.rotating:
            T = self.solve_torques()
            T += self.random_torque()

            if self.isotropic:
                w1 = self.alpha_R*T
            else:
                w1 = np.einsum('Nij,Nj->Ni', self.alpha_R_rot, T)

            self.solver.angular_velocity = w1
            w1_q = np.array([np.quaternion(*omega) for omega in w1])
            self.solver.orientation = (1 + w1_q*self.dt/2)*self.solver.orientation

    def ld_step(self):
        if self.solver.rotating:
            raise NotImplementedError('ld step with rotation')

        if not self.isotropic:
            raise NotImplementedError('ld step with anisotropic particles')

        F = self.solve_forces()
        Fr = self.random_force()
        F += Fr

        mass = self.solver.mass[:,np.newaxis]
        v0 = self.solver.velocity
        v1 = self.alpha_T*F + (v0 - self.alpha_T*F)*np.exp(-self.dt/mass/self.alpha_T)

        dr = self.alpha_T*F*self.dt - self.alpha_T*mass*(self.alpha_T*F - v0)*(1 - np.exp(-self.dt/mass/self.alpha_T))

        self.solver.velocity = v1
        self.solver.position += dr

    def hbd_step(self):
        if not self.isotropic:
            raise NotImplementedError('hbd step with anisotropic particles')

        F = self.solve_forces()
        T = self.solve_torques()

        grand_F = np.hstack([F.flatten(), T.flatten()]) 
        grand_noise = np.random.normal(size=grand_F.shape)

        Np = self.solver.Nparticles
        grand_velocity = self.grand_M @ grand_F + self.grand_N @ grand_noise
        v1 = grand_velocity[:3*Np].reshape([Np,3])
        w1 = grand_velocity[3*Np:].reshape([Np,3])

        self.solver.velocity = v1
        self.solver.position += self.dt*v1

        self.solver.angular_velocity = w1
        w1_q = np.array([np.quaternion(*omega) for omega in w1])
        self.solver.orientation = (1 + w1_q*self.dt/2)*self.solver.orientation
