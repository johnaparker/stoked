import matplotlib.pyplot as plt
import numpy as np

import stoked

us = 1e-6
nm = 1e-9


def torque(time, position, orientation):
    """External, constant z-torque on all particles."""
    T = np.zeros_like(position)
    T[:, 2] = 4e-18
    return T


initial = stoked.hexagonal_lattice_particles(7) * 600 * nm

np.random.seed(0)
sim = stoked.stokesian_dynamics(
    position=initial,
    temperature=300,
    drag=stoked.drag_sphere(75e-9, 8e-4),
    dt=1 * us,
    constraint=stoked.constrain_position(z=0),
    torque=torque,
)
trajectory = sim.run(5000)
trajectory.position /= nm

fig, ax = plt.subplots()

anim = stoked.trajectory_animation(trajectory[::2], patches=stoked.circle_patches(75))
ax.set(xlabel="x (nm)", ylabel="y (nm)")

plt.show()
