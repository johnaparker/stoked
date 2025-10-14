import matplotlib.pyplot as plt
import numpy as np

import stoked

us = 1e-6
nm = 1e-9


def torque(time, position, orientation):
    """External, constant z-torque on all particles."""
    T = np.zeros_like(position)
    T[:, 2] = 2e-21
    return T


def force(time, position, orientation):
    """External, constant z-torque on all particles."""
    F = np.zeros_like(position)
    F[:, 0] = 2e-13
    return F


np.random.seed(0)

Nsteps = 10000
rx = rz = 25 * nm
ry = 100 * nm
sim = stoked.brownian_dynamics(
    position=[0, 0, 0],
    temperature=0,
    drag=stoked.drag_ellipsoid([rx, ry, rz], 8e-4),
    dt=1 * us,
    constraint=[stoked.constrain_position(z=0), stoked.constrain_rotation(x=True, y=True)],
    torque=torque,
    force=force,
)

sim.rotating = True
trajectory = sim.run(Nsteps)
trajectory.position /= nm
trajectory.position = trajectory.position[::50]
trajectory.orientation = trajectory.orientation[::50]

fig, ax = plt.subplots()

anim = stoked.trajectory_animation(
    trajectory, patches=stoked.ellipse_patches(rx / nm, ry / nm), trail=Nsteps, trail_type="solid"
)
ax.set(xlabel="x (nm)", ylabel="y (nm)")

plt.show()
