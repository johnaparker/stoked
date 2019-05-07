from stoked import stokesian_dynamics, trajectory_animation, drag_sphere
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
history = np.zeros([Nsteps,len(position),3], dtype=float)
wz = np.zeros([Nsteps,len(position)], dtype=float)

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
for i in tqdm(range(Nsteps)):
    history[i] = sim.position.squeeze()
    wz[i] = sim.angular_velocity[...,2]
    sim.step()


fig, ax = plt.subplots()
angles = np.zeros_like(wz)
for i in range(len(position)):
    angles[:,i] = cumtrapz(wz[:,i], np.arange(Nsteps)*dt, initial=0)
anim = trajectory_animation(history[::1]/1e-9, radii=75, projection='z', interval=30, angles=angles[::1])
# save_animation(anim, 'out.mp4')

plt.show()
