from stoked import brownian_dynamics, trajectory_animation, drag_sphere
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.constants import k as kb

no_force = None
def harmonic_force(t, rvec, orientation, k=1):
    return -k*rvec

position = [0,0,0]
drag = drag_sphere(1/(6*np.pi), 1)
temperature = .05
dt = 10

Nsteps = 10000
fig, ax = plt.subplots()
ax.set_aspect('equal')

for Fext in [no_force, harmonic_force]:
    history = np.zeros([Nsteps,3], dtype=float)

    if Fext is no_force:
        sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, force=Fext)
    else:
        sim = brownian_dynamics(position=position, drag=drag, temperature=temperature, dt=dt, force=partial(Fext, k=0.01))

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        sim.step()

    ax.plot(history[:,0], history[:,1], lw=.5)
    ax.legend()

    if Fext is harmonic_force:
        fig, ax = plt.subplots()
        ax.hexbin(history[:,0], history[:,1])
        ax.set_aspect('equal')

        fig, ax = plt.subplots()
        rad = np.linalg.norm(history, axis=1)
        hist, edges = np.histogram(rad, bins=100)

        rad = edges[1:]
        counts = hist/(4*np.pi*rad**2)
        E = -kb*temperature*np.log(counts)
        E -= E[6]
        ax.plot(rad, E)
        ax.plot(rad, 0.5*.01*rad**2, 'o', ms=3)


fig, ax = plt.subplots()
anim = trajectory_animation(history.reshape([Nsteps,1,3]), radii=1e-11, projection='z', interval=30)

plt.show()


