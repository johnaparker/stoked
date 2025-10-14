from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from stoked import brownian_dynamics, circle_patches, drag_sphere, trajectory_animation


def bigaussian_force(t, rvec, orientation, rvec1, rvec2, A1=1, w1=1, A2=1, w2=1):
    r1 = np.linalg.norm(rvec - rvec1, axis=1)[:, np.newaxis]
    r2 = np.linalg.norm(rvec - rvec2, axis=1)[:, np.newaxis]
    F1 = -A1 * (rvec - rvec1) / w1**2 * np.exp(-(r1**2) / (2 * w1**2))
    F2 = -A2 * (rvec - rvec2) / w2**2 * np.exp(-(r2**2) / (2 * w2**2))

    return F1 + F2


delta = 1e-10
r1 = [-delta * 0.9, 0, 0]
r2 = [delta * 0.9, 0, 0]
w1 = w2 = delta / 1.3
A1 = A2 = 1e-22
position = [-delta, 0, 0]
drag = drag_sphere(1 / (6 * np.pi), 1)
temperature = 0.4
dt = 10

Nsteps = 100000
fig, ax = plt.subplots()
ax.set_aspect("equal")

history = np.zeros([Nsteps, 3], dtype=float)
sim = brownian_dynamics(
    position=position,
    drag=drag,
    temperature=temperature,
    dt=dt,
    force=partial(bigaussian_force, rvec1=r1, rvec2=r2, A1=A1, A2=A2, w1=w1, w2=w2),
)

for i in tqdm(range(Nsteps)):
    history[i] = sim.position.squeeze()
    sim.step()

ax.plot(history[:, 0], history[:, 1], lw=0.5)

fig, ax = plt.subplots()
ax.hexbin(history[:, 0], history[:, 1], extent=[-1.5 * delta, 1.5 * delta, -delta, delta])
ax.set_aspect("equal")

fig, ax = plt.subplots()
ax.plot(history[:, 0])

fig, ax = plt.subplots()
anim = trajectory_animation(history.reshape([Nsteps, 1, 3]), patches=circle_patches(1e-11), interval=30)

plt.show()
