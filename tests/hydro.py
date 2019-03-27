from pedesis import brownian_dynamics, trajectory_animation, drag_sphere
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from topics.photonic_clusters.create_lattice import hexagonal_lattice_particles

position = [[0,0,0], [600e-9,0,0]]
position = 600e-9*hexagonal_lattice_particles(7)
drag = drag_sphere(75e-9, .6e-3)
temperature = 90
dt = 100e-9

Nsteps = 1000
history = np.zeros([Nsteps,len(position),3], dtype=float)

def Fext(t, rvec, orientation):
    F = np.zeros_like(rvec)
    A = 1e-10
    omega = 400000
    F[0,1] = A*np.cos(omega*t)

    return F

def torque(t, rvec, orientation):
    T = np.zeros_like(rvec)
    T[:,2] = 1e-16
    return T

sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, hydrodynamic_coupling=True,
        torque=torque)
for i in tqdm(range(Nsteps)):
    history[i] = sim.position.squeeze()
    sim.step()


fig, ax = plt.subplots()
anim = trajectory_animation(history[::10], radii=75e-9, projection='z', interval=30)

plt.show()
