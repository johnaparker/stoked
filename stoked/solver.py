import numpy as np
from scipy.constants import k as kb
from collections.abc import Iterable
import quaternion
from abc import ABCMeta, abstractmethod
from .hydrodynamics import grand_mobility_matrix
from tqdm import tqdm

class trajectory:
    def __init__(self, position, orientation=None):
        self.position = np.asarray(position, dtype=float)

        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=np.quaternion)
        else:
            self.orientation = None

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

    def _update(self, time, position, orientation, temperature):
        self.time = time
        self.position = position
        self.orientation = orientation
        self.temperature = temperature
        self.update()

def fluctuation(friction, T, dt):
    """Given a friction matrix, return the fluctuation matrix"""
    rhs = 2*kb*T*friction/dt
    return np.linalg.cholesky(rhs)

class stokesian_dynamics:
    """
    Perform a Stokesian dynamics simulation, with optional external and internal internal interactions and rotational dynamics
    """
    def __init__(self, *, temperature, dt, position, drag, orientation=None, 
                 force=None, torque=None, interactions=None, constraint=None,
                 hydrodynamic_coupling=True):
        """
        Arguments:
            temperature        system temperature
            dt                 time-step
            position[N,D]      initial position of N particles in D dimensions
            drag               drag coefficients (of base type stoked.drag)
            orientation[N]     initial orientation of N particles (as quaternions)
            force(t, r, q)     external force function given time t, position r[N,D], orientation q[N] and returns force F[N,D] (can be a list of functions)
            torque(t, r, q)    external torque function given time t, position r[N,D], orientation q[N] and returns torque T[N,D] (can be a list of functions)
            interactions       particle interactions (can be a list)
            constraint         particle constraints (can be a list)
            hydrodynamic_coupling    if True, include hydrodynamic coupling interactions (default: True)
        """
        self.temperature = temperature
        self.dt = dt
        self.time = 0
        self.position = np.atleast_2d(np.asarray(position, dtype=float))
        self.drag = drag
        self.Nparticles = len(self.position)
        self.ndim = self.position.shape[1]
        self.hydrodynamic_coupling = hydrodynamic_coupling

        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=np.quaternion)
        else:
            self.orientation = np.ones([self.Nparticles], dtype=np.quaternion)

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

        if constraint is None:
            self.constraint = []
        elif not isinstance(constraint, Iterable):
            self.constraint = [constraint]
        else:
            self.constraint = constraint

        self.alpha_T = np.zeros_like(self.position)
        drag_T = self.drag.drag_T
        if self.drag.isotropic and not np.isscalar(drag_T):
            drag_T = drag_T[:,np.newaxis]

        self.alpha_T[...] = 1/drag_T
        self.velocity = np.zeros_like(self.position)

        if self.orientation is not None:
            self.alpha_R = np.zeros_like(self.position)
            drag_R = self.drag.drag_R
            if self.drag.isotropic and not np.isscalar(drag_R):
                drag_R = drag_R[:,np.newaxis]

            self.alpha_R[...] = 1/drag_R
            self.angular_velocity = np.zeros_like(self.position)

        self.isotropic = self.drag.isotropic and not self.hydrodynamic_coupling
        if self.isotropic and orientation is None:
            self.rotating = False
        else:
            self.rotating = True

    def step(self):
        """
        Time-step the positions (and orientations) by dt
        """
        if self.hydrodynamic_coupling:
            self._step_hydrodynamics()
        else:
            self._step()

    def run(self, Nsteps):
        """
        Run the simulation for Nsteps, returning the trajectories
        """
        position = np.zeros((Nsteps,) + self.position.shape, dtype=float)

        if self.rotating:
            orientation = np.zeros((Nsteps,) + self.orientation.shape, dtype=np.quaternion)
        else:
            orientation = None

        for i in tqdm(range(Nsteps), desc='Running dynamics'):
            position[i] = self.position
            if self.rotating:
                orientation[i] = self.orientation
            self.step()

        return trajectory(position, orientation)

    def run_until(self, condition):
        """
        Run the simulation until some condition is met, returning the trajectories
        """
        position = []

        if self.rotating:
            orientation = []
        else:
            orientation = None

        with tqdm(desc='Running dynamics until condition is met') as pbar:
            while not condition():
                position.append(np.copy(self.position))
                if self.rotating:
                    orientation.append(np.copy(self.orientation))
                self.step()
                pbar.update()

        position = np.array(position, dtype=float)
        if self.rotating:
            orientation = np.array(orientation, dtype=np.quaternion)

        return trajectory(position, orientation)

    def _step(self):
        self._update_interactions(self.time, self.position, self.orientation)

        F = self._total_force(self.time, self.position, self.orientation)
        noise_T = np.random.normal(size=self.position.shape) 
        v1 = self._get_velocity(self.alpha_T, F, noise_T, self.orientation)
        r_predict = self.position + self.dt*v1

        if self.rotating:
            T = self._total_torque(self.time, self.position, self.orientation)
            noise_R = np.random.normal(size=self.position.shape) 
            w1 = self._get_velocity(self.alpha_R, T, noise_R, self.orientation)
            w1_q = np.array([np.quaternion(*omega) for omega in w1])
            o_predict = (1 + w1_q*self.dt/2)*self.orientation
        else:
            o_predict = self.orientation

        self.time += self.dt
        self._perform_constraints(r_predict, o_predict)
        self._update_interactions(self.time, r_predict, o_predict)

        F_predict = self._total_force(self.time, r_predict, o_predict)
        v2 = self._get_velocity(self.alpha_T, F_predict, noise_T, o_predict)
        self.velocity = 0.5*(v1 + v2)
        self.position += self.dt*self.velocity

        if self.rotating:
            T_predict =  self._total_torque(self.time, r_predict, o_predict)
            w2 = self._get_velocity(self.alpha_R, T_predict, noise_R, o_predict)
            w2_q = np.array([np.quaternion(*omega) for omega in w2])
            w_q = (w1_q + w2_q)/2
            self.angular_velocity = 0.5*(w1 + w2)
            self.orientation = (1 + w_q*self.dt/2)*self.orientation

        self._perform_constraints(self.position, self.orientation)
        if self.rotating:
            self.orientation = np.normalized(self.orientation)

    def _perform_constraints(self, position, orientation):
        for constraint in self.constraint:
            constraint(position, orientation)

    def _step_hydrodynamics(self):
        self._update_interactions(self.time, self.position, self.orientation)

        F = self._total_force(self.time, self.position, self.orientation)
        noise_T = np.random.normal(size=self.position.shape) 

        T = self._total_torque(self.time, self.position, self.orientation)
        noise_R = np.random.normal(size=self.position.shape) 

        v1, w1 = self._get_velocity_hydrodynamics(F, T, noise_T, noise_R, self.orientation)

        r_predict = self.position + self.dt*v1
        w1_q = np.array([np.quaternion(*omega) for omega in w1])
        o_predict = (1 + w1_q*self.dt/2)*self.orientation


        self.time += self.dt
        self._update_interactions(self.time, r_predict, o_predict)

        F_predict = self._total_force(self.time, r_predict, o_predict)
        T_predict =  self._total_torque(self.time, r_predict, o_predict)

        v2, w2 = self._get_velocity_hydrodynamics(F_predict, T_predict, noise_T, noise_R, o_predict)

        w2_q = np.array([np.quaternion(*omega) for omega in w2])
        w_q = (w1_q + w2_q)/2

        self.velocity = 0.5*(v1 + v2)
        self.position += self.dt*self.velocity
        self.angular_velocity = 0.5*(w1 + w2)
        self.orientation = (1 + w_q*self.dt/2)*self.orientation

        self.orientation = np.normalized(self.orientation)

    def _get_velocity(self, alpha, drive, noise, orientation):
        """
        FOR INTERNAL USE ONLY

        obtain velocity (or angular velocity) given alpha parameter, drive force/torque, noise, and orientation
        """
        if self.isotropic:
            beta = np.sqrt(2*kb*self.temperature*alpha/self.dt)
            velocity = alpha*drive + beta*noise
        else:
            rot = quaternion.as_rotation_matrix(orientation)
            alpha = np.einsum('Nij,Nj,Nlj->Nil', rot, alpha, rot)

            beta = np.zeros_like(alpha)
            for i in range(self.Nparticles):
                beta[i] = fluctuation(alpha[i], self.temperature, self.dt)

            velocity = np.einsum('Nij,Nj->Ni', alpha, drive) + np.einsum('Nij,Nj->Ni', beta, noise)

        return velocity

    def _get_velocity_hydrodynamics(self, F, T, noise_T, noise_R, orientation):
        """
        FOR INTERNAL USE ONLY
        """
        drag_T = np.zeros([self.Nparticles, 3, 3], dtype=float)
        drag_R = np.zeros([self.Nparticles, 3, 3], dtype=float)
        if self.drag.isotropic:
            np.einsum('Nii->Ni', drag_T)[...] = self.alpha_T
            np.einsum('Nii->Ni', drag_R)[...] = self.alpha_R
        else:
            rot = quaternion.as_rotation_matrix(orientation)
            drag_T[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_T, rot)
            drag_R[...] = np.einsum('Nij,Nj,Nlj->Nil', rot, self.alpha_R, rot)

        M = grand_mobility_matrix(self.position, drag_T, drag_R, self.drag.viscosity)

        grand_F = np.hstack([F.flatten(), T.flatten()]) 
        grand_noise = np.hstack([noise_T.flatten(), noise_R.flatten()]) 
        N = fluctuation(M, self.temperature, self.dt)

        grand_velocity = M @ grand_F + N @ grand_noise
        velocity = grand_velocity[:3*self.Nparticles].reshape([self.Nparticles,3])
        angular_velocity = grand_velocity[3*self.Nparticles:].reshape([self.Nparticles,3])

        return velocity, angular_velocity

    def _update_interactions(self, time, position, orientation):
        """
        FOR INTERNAL USE ONLY

        update the interactions given a new (or predicted) position and orientation
        """
        if self.interactions is not None:
            for I in self.interactions:
                I._update(time, position, orientation, self.temperature)

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

def brownian_dynamics(*, temperature, dt, position, drag, orientation=None, 
                 force=None, torque=None, interactions=None, constraint=None):
    """
    Perform a Brownian dynamics simulation, with optional external and internal internal interactions and rotational dynamics
        
    Arguments:
        temperature        system temperature
        dt                 time-step
        position[N,D]      initial position of N particles in D dimensions
        drag               drag coefficients (of base type stoked.drag)
        orientation[N]     initial orientation of N particles (as quaternions)
        force(t, r, q)     external force function given time t, position r[N,D], orientation q[N] and returns force F[N,D] (can be a list of functions)
        torque(t, r, q)    external torque function given time t, position r[N,D], orientation q[N] and returns torque T[N,D] (can be a list of functions)
        interactions       particle interactions (can be a list)
        constraint         particle constraints (can be a list)
    """
    return stokesian_dynamics(temperature=temperature, dt=dt, position=position, drag=drag,
                  orientation=orientation, force=force, torque=torque, interactions=interactions,
                  constraint=constraint, hydrodynamic_coupling=False)
