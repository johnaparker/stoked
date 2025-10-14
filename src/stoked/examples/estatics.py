from functools import partial

import matplotlib.pyplot as plt
import numpy as np

import stoked


def harmonic_force(time, position, orientation, stiffness):
    return -stiffness * position


nm = 1e-9
us = 1e-6

stiffness = 2e-6
radius = 25 * nm

N = 15
initial = np.random.uniform(-300 * nm, 300 * nm, size=(N, 2))
Q = 1e-18
bd = stoked.brownian_dynamics(
    position=initial,
    temperature=300,
    drag=stoked.drag_sphere(radius=radius, viscosity=8e-4),
    dt=0.2 * us,
    force=partial(harmonic_force, stiffness=stiffness),
    interactions=stoked.electrostatics(Q),
)

trajectory = bd.run(10000).position

fig, ax = plt.subplots()
ax.plot(trajectory[..., 0] / nm, trajectory[..., 1] / nm, lw=0.5)
ax.set(aspect="equal", xlabel="x (nm)", ylabel="y (nm)")
plt.show()
