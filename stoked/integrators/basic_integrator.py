from . import integrator
import numpy as np

class basic_integrator(integrator):
    def __init__(self, grand_mobility_interval=10):
        super().__init__(grand_mobility_interval)

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
            self.solver.angular_velocity = w1
            w1_q = np.array([np.quaternion(*omega) for omega in w1])
            self.solver.orientation = (1 + w1_q*self.dt/2)*self.solver.orientation

            # rot = quaternion.as_rotation_matrix(orientation)
            # alpha = np.einsum('Nij,Nj,Nlj->Nil', rot, alpha, rot)

            # beta = np.zeros_like(alpha)
            # for i in range(self.Nparticles):
                # beta[i] = fluctuation(alpha[i], self.temperature, self.dt)

            # velocity = np.einsum('Nij,Nj->Ni', alpha, drive) + np.einsum('Nij,Nj->Ni', beta, noise)

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

    def hbd_step(self):
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
