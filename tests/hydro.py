from stoked import stokesian_dynamics, trajectory_animation, drag_sphere, circle_patches
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from topics.photonic_clusters.create_lattice import hexagonal_lattice_particles
from my_pytools.my_matplotlib.animation import save_animation
from scipy.integrate import cumtrapz

position = [[0,0,0], [600e-9,0,0]]
position = 600e-9*hexagonal_lattice_particles(7)
drag = drag_sphere(75e-9, .6e-3)
temperature = 9000
dt = 100e-9

Nsteps = 1000

def Fext(t, rvec, orientation):
    F = np.zeros_like(rvec)
    A = 1e-10
    omega = 400000
    F[0,1] = A*np.cos(omega*t)

    return F

def torque(t, rvec, orientation):
    T = np.zeros_like(rvec)
    T[:,2] = 8e-17
    return T

sim = stokesian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, torque=torque)
history = sim.run(Nsteps)
history.position *= 1e9

fig, ax = plt.subplots()
anim = trajectory_animation(history, patches=circle_patches(75))
# save_animation(anim, 'out.mp4')

plt.show()
