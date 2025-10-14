from collections.abc import Iterable
from copy import deepcopy
from itertools import cycle

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import stoked


def trajectory_snapshots(trajectory, patches=None, N=5, colors=None, circle_kwargs=None, dt=None, time_unit="s"):
    """Create N shapshots of a trajectory.

    Arguments:
        trajectory     trajectory data for T steps, N particles
        patches        patches object to represent particles
    """
    if circle_kwargs is None:
        circle_kwargs = {}
    if isinstance(N, Iterable):
        Nrow, Ncol = N
    else:
        Nrow, Ncol = 1, N
    Ntot = Nrow * Ncol
    fig, axes = plt.subplots(
        ncols=Ncol, nrows=Nrow, figsize=(3 * Ncol, 2.7 * Nrow), sharex=True, sharey=True, constrained_layout=True
    )

    circle_properties = dict(linewidth=3, zorder=1, fill=False)
    circle_properties.update(circle_kwargs)
    line_properties = deepcopy(circle_properties)

    if colors is None:
        colors = mcolors.TABLEAU_COLORS
    if isinstance(colors, str):
        colors = [colors]
    if not isinstance(colors, Iterable):
        colors = [colors]
    color_cycle = cycle(colors)

    if not isinstance(trajectory, stoked.trajectory):
        trajectory = stoked.trajectory(trajectory)

    if trajectory.orientation is not None:
        angles = stoked.quaternion_to_angles(trajectory.orientation)
    else:
        angles = [None] * len(trajectory.position)

    trajectory.position.shape[0]
    Nparticles = trajectory.position.shape[1]

    if patches is not None:
        if isinstance(patches, list):
            patches_type = [p.patches_type for p in patches]
            patches_args = [p.args for p in patches]
        else:
            if not isinstance(patches.patches_type, Iterable):
                patches_type = [patches.patches_type] * Nparticles
            else:
                patches_type = patches.patches_type

            if not isinstance(patches.args[0], Iterable):
                patches_args = [patches.args for i in range(Nparticles)]
            else:
                patches_args = [[arg[i] for arg in patches.args] for i in range(Nparticles)]

    for j in range(Ntot):
        ax = axes.flatten()[j]
        inv = ax.transData.transform
        for i in range(Nparticles):
            Ti = (len(trajectory.position) - 1) // (Ntot - 1) * j
            pos = trajectory.position[Ti, i]
            color = next(color_cycle)

            if patches is not None:
                if isinstance(patches_type[i], mpl.collections.PatchCollection):
                    ax.add_collection(patches_type[i])
                    patches_type[i].set(facecolor=color, edgecolor="k")
                else:
                    patch = patches_type[i](pos, *patches_args[i], fc=color, edgecolor=color, **circle_properties)
                    ax.add_patch(patch)

                if patches_type[i] is plt.Circle and angles[Ti] is not None:
                    radius = patches_args[i][0]
                    line = plt.Line2D(
                        [pos[0] - radius, pos[0] + radius],
                        [pos[1], pos[1]],
                        lw=circle_properties["linewidth"],
                        color=color,
                        **line_properties,
                    )
                    ax.add_line(line)
            else:
                dot = plt.Circle(inv(pos[:2]), 3, color=color, transform=None)
                ax.add_patch(dot)

        ax.autoscale()
        ax.set_aspect("equal")

        if dt is not None:
            ax.text(
                0.98,
                0.02,
                rf"{dt * Ti:.2f} {time_unit}",
                transform=ax.transAxes,
                horizontalalignment="right",
            )

    return fig, axes
