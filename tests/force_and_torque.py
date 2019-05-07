from stoked import stokesian_dynamics, ellipsoid_drag, trajectory_animation
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.integrate import cumtrapz

position = [0,0,0]
drag = ellipsoid_drag([1,1.5,1], 1)
temperature = .00001
dt = 10

Nsteps = 1000

history = np.zeros([Nsteps,3], dtype=float)
omega_z = np.zeros([Nsteps], dtype=float)

def force(t, pos, orient):
    return np.array([[2e-12,0,0]])

def torque(t, pos, orient):
    return np.array([[0,0,5e-3]])

sim = stokesian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt,
                        force=force, torque=torque)

for i in tqdm(range(Nsteps)):
    history[i] = sim.position.squeeze()
    omega_z[i] = sim.angular_velocity[0,2]
    sim.step()

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.plot(history[:,0], history[:,1], lw=.5)
ax.legend()

fig, ax = plt.subplots()
time = np.arange(Nsteps)*dt
ax.plot(time, cumtrapz(omega_z, time, initial=0))

fig, ax = plt.subplots()
anim = trajectory_animation(history.reshape([Nsteps,1,3])[::2], radii=1e-11, projection='z', interval=30, trail=500, trail_type='fading')

plt.show()
