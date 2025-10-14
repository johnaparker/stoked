import matplotlib.pyplot as plt
import numpy as np

import stoked

us = 1e-6
nm = 1e-9


def force(time, position, orientation):
    """An oscillating force on particle 0."""
    F = np.zeros_like(position)
    A = 2e-10
    omega = 800000
    F[0, 1] = A * np.cos(omega * time)

    return F


### create a chain of particles
xvals = np.linspace(0, 1700 * nm, 8)
initial = [(x, 0, 0) for x in xvals]  # now in 3 dimensions

bd = stoked.stokesian_dynamics(
    position=initial,
    temperature=300,
    drag=stoked.drag_sphere(75e-9, 8e-4),
    dt=0.1 * us,
    constraint=stoked.constrain_position(z=0),  # constrain the particles to z = 0
    interactions=stoked.collisions_sphere(75e-9, 30),
    force=force,
)

bd.rotating = True  # perform rotational dynamics
trajectory = bd.run(5000)
trajectory.position /= nm  # convert the position data to nano-meters

fig, ax = plt.subplots()

anim = stoked.trajectory_animation(trajectory[::10], patches=stoked.circle_patches(75))
ax.set(xlabel="x (nm)", ylabel="y (nm)")
plt.show()
