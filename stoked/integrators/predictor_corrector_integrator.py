from . import integrator
import numpy as np

class predictor_corrector_integrator(integrator):
    def __init__(self, grand_mobility_interval=10):
        super().__init__(grand_mobility_interval)

    def bd_step(self):
        r0 = np.copy(self.solver.position)
        o0 = np.copy(self.solver.orientation)

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

            w0 = np.copy(self.solver.angular_velocity)
            o0 = np.copy(self.solver.orientation)
            self.solver.angular_velocity = w1
            self.solver.orientation = (1 + w1_q*self.dt/2)*self.solver.orientation

        self.solver.time += self.dt
        self.perform_constraints()
        self.pre_step()

        F = self.solve_forces()
        F += Fr
        v2 = self.alpha_T*F
        self.solver.velocity = (v1 + v2)/2
        self.solver.position = r0 + self.dt*self.solver.velocity

        if self.solver.rotating:
            T = self.solve_torques()
            w2 = self.alpha_R*T
            w2_q = np.array([np.quaternion(*omega) for omega in w1])
            w_q = (w1_q + w2_q)/2

            self.solver.angular_velocity = (w1 + w2)/2
            self.solver.orientation = (1 + w1_q*self.dt/2)*o0

        self.solver.time -= self.dt
