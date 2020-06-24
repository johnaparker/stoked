import numpy as np
from collections.abc import Iterable
import quaternion
from abc import ABCMeta, abstractmethod
from .hydrodynamics import grand_mobility_matrix, particle_wall_self_mobility
from .integrators import basic_integrator
from tqdm import tqdm

class trajectory:
    def __init__(self, position, orientation=None):
        self.position = np.asarray(position, dtype=float)

        if orientation is not None:
            self.orientation = np.asarray(orientation, dtype=np.quaternion)
        else:
            self.orientation = None

    def __getitem__(self, key):
        orientation_s = None if self.orientation is None else self.orientation[key]
        return trajectory(self.position[key], orientation_s)

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

class stokesian_dynamics:
    """
    Perform a Stokesian dynamics simulation, with optional external and internal interactions and rotational dynamics
    """
    def __init__(self, *, temperature, dt, position, drag, orientation=None, 
                 force=None, torque=None, interactions=None, constraint=None,
                 interface=None, inertia=None, hydrodynamic_coupling=True, integrator=None):
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
            interface          no-slip boundary interface (default: no interface)
            inertia            inertia coefficients of base type stoked.inertia (default: over-damped dynamics)
            hydrodynamic_coupling    if True, include hydrodynamic coupling interactions (default: True)
            integrator         dynamics integrator
        """
        self.temperature = temperature
        self.dt = dt
        self.time = 0.0
        self.position = np.atleast_2d(np.asarray(position, dtype=float))
        self.drag = drag
        self.Nparticles = len(self.position)
        self.ndim = self.position.shape[1]
        self.interface = interface
        self.inertia = inertia
        self.hydrodynamic_coupling = hydrodynamic_coupling
        self.integrator = integrator

        if integrator is None:
            self.integrator = basic_integrator()

        if self.interface is not None and self.ndim != 3:
            raise ValueError('An interface can only be used in 3-dimensions')

        if not self.drag.isotropic and self.interface is not None:
            raise NotImplementedError('anisotropic particles with an interface')

        if orientation is not None:
            self.orientation = np.atleast_1d(np.asarray(orientation, dtype=np.quaternion))
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
        self._combine_pairwise_interactions()

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

        if self.inertia is not None:
            self.mass = np.empty([self.Nparticles], dtype=float)
            self.moment = np.empty([self.Nparticles,3], dtype=float)

            self.mass[...] = self.inertia.mass
            self.moment[...] = self.inertia.moment

        self._init_integrator()
        self._num_thermal_steps = 1

    def step(self):
        """
        Time-step the positions (and orientations) by dt
        """
        self.integrator.pre_step()
        for n in range(self._num_thermal_steps):
            self.step_func()
            self.integrator.post_step()

    def run(self, Nsteps, progress=True):
        """
        Run the simulation for Nsteps, returning the trajectories

        Arguments:
            Nsteps       number of steps to run
            progress     If True, display a progress bar
        """
        position = np.zeros((Nsteps,) + self.position.shape, dtype=float)

        if self.rotating:
            orientation = np.zeros((Nsteps,) + self.orientation.shape, dtype=np.quaternion)
        else:
            orientation = None

        for i in tqdm(range(Nsteps), desc='Running dynamics', disable=(not progress)):
            position[i] = self.position
            if self.rotating:
                orientation[i] = self.orientation
            self.step()

        return trajectory(position, orientation)

    def run_until(self, condition):
        """
        Run the simulation until some condition is met, returning the trajectories

        Arguments:
            condition    Function to return True when simulation should terminate
            progress     If True, display a progress bar
        """
        position = []

        if self.rotating:
            orientation = []
        else:
            orientation = None

        with tqdm(desc='Running dynamics until condition is met', disable=(not progress)) as pbar:
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

    def _combine_pairwise_interactions(self):
        from stoked.forces import pairwise_central_force

        if self.interactions is not None:
            pairwise_forces = []
            for i,I in enumerate(self.interactions):
                if isinstance(I, pairwise_central_force):
                    pairwise_forces.append(I)

            if len(pairwise_forces) > 1:
                for pf in pairwise_forces:
                    self.interactions.remove(pf)
                self.interactions.append(sum(pairwise_forces[1:], pairwise_forces[0]))

    def _init_integrator(self):
        self.integrator.initialize(self)

        if self.hydrodynamic_coupling:
            if self.inertia is None:
                self.step_func = self.integrator.hbd_step
            else:
                self.step_func = self.integrator.hld_step
        else:
            if self.inertia is None:
                self.step_func = self.integrator.bd_step
            else:
                self.step_func = self.integrator.ld_step

def brownian_dynamics(*, temperature, dt, position, drag, orientation=None, 
                 force=None, torque=None, interactions=None, constraint=None, interface=None, inertia=None, integrator=None):
    """
    Perform a Brownian dynamics simulation, with optional external and internal interactions and rotational dynamics
        
    Arguments:
        temperature        system temperature
        dt                 time-step
        position[N,D]      initial position of N particles in D dimensions
        drag               drag coefficients (of base type stoked.drag)
        orientation[N]     initial orientation of N particles (as quaternions)
        force(t, r, q)     external force function given time t, position r[N,D], orientation q[N] and returns force F[N,D] (can be a list of functions)
        torque(t, r, q)    external torque function given time t, position r[N,D], orientation q[N] and returns torque T[N,D] (can be a list of functions)
        interactions       particle interactions (can be a list)
        interface          no-slip boundary interface (default: no interface)
        constraint         particle constraints (can be a list)
        inertia            inertia coefficients of base type stoked.inertia (default: over-damped dynamics)
        integrator         dynamics integrator
    """
    return stokesian_dynamics(temperature=temperature, dt=dt, position=position, drag=drag,
                  orientation=orientation, force=force, torque=torque, interactions=interactions,
                  constraint=constraint, interface=interface, inertia=inertia, hydrodynamic_coupling=False,
                  integrator=integrator)
