from itertools import cycle
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from .colored_plot import colored_plot
import matplotlib.patheffects as path_effects
from tqdm import tqdm
import matplotlib.colors as mcolors
from collections.abc import Iterable
from collections import namedtuple
import stoked

patches = namedtuple('patches', ['patches_type', 'args'])
def circle_patches(radius):
    return patches(plt.Circle, (radius,))

def ellipse_patches(rx, ry):
    return patches(mpl.patches.Ellipse, (rx, ry))

def atleast(array, dim, length, dtype=None):
    """Given an n-dimensional array, return either an n or n+1 dimensional repeated array
        array      input array (or scalar)
        dim        dimension of the returned array (must be n or n+1)
        dtype      array datatype (default None)
    """

    ret = np.asarray(np.atleast_1d(array), dtype=dtype)
    if (dim not in (ret.ndim,ret.ndim+1)):
        raise ValueError('dim = {0} is invalid with input dim = {1}'.format(dim, ret.ndim))

    if len(ret.shape) == dim and ret.shape[0] != length:
        ret = np.repeat(ret, length)
    elif len(ret.shape) != dim:
        ret = np.array([ret]*length)

    return ret

def rotation_transform(axis, angle, ax = None):
    """Return a rotation transfrom that rotates around an axis by an angle
            axix[2]       (x,y) center of rotation
            angle         angle of rotation (in degrees)  """

    if ax is None: ax = plt.gca()
    t_scale = ax.transData
    t_rotate = mpl.transforms.Affine2D().rotate_deg_around(axis[0], axis[1], angle*180/np.pi)
    return t_rotate + t_scale

#TODO use matplotlib collections, patchcollection instead of python lists for performance
def animation_2d(func, patches, frames=None, colors=None, ax=None, time=None, time_unit='T',
        xlim=None, ylim=None, number_labels=False, trail=100, trail_type='fading',
        time_kwargs={}, label_kwargs={}, circle_kwargs={}, trail_kwargs={}, fading_kwargs={}, **kwargs):
    """Create a 2D animation of trajectories

       Arguments:
            func                       function that returns (x,y) coordinates and angles
            colors                     list of colors to cycle through
            ax (default None)          specify the axes of the animation
            time[N]                    display the time (in time_units)
            xlim[2]                    min,max values of x-axis
            ylim[2]                    min,max values of y-axis
            time_unit                  string label for the units of time (default 'T')
            number_labels              include text labels (1,2,...) per particle
            trail                      length of particle trail
            trail_type                 'normal' or 'fading'
            time_kwargs                additional arguments to pass to timer text object
            label_kwargs               additional arguments to pass to label text objects
            circle_kwargs              additional arguments to circle properites
            trail_kwargs               additional arguments to line trail properies
            fading_kwargs              Fading line properites, {max_lw, min_lw}
            kwargs                     Additional kwargs for FuncAnimation
    """
    trail_types = ['normal', 'fading']
    if trail_type not in trail_types:
        raise ValueError("trail_type '{}' is not valid. Choose from {}".format(trail_type, trail_types))
    if (trail_type == 'fading' and trail == np.inf):
        raise ValueError("trail cannot be fading and infinite")

    if colors is None:
        colors = mcolors.TABLEAU_COLORS
    if isinstance(colors, str):
        colors = [colors]
    if not isinstance(colors, Iterable):
        colors = [colors]

    time_properties = dict(fontsize=12, zorder=2)
    time_properties.update(time_kwargs)

    label_properties = dict(horizontalalignment='center', verticalalignment='center', fontsize=14, zorder=2)
    label_properties.update(label_kwargs)

    trail_properties = dict(zorder=0)
    trail_properties.update(trail_kwargs)

    circle_properties = dict(facecolor=(1,1,1,0), linewidth=2, zorder=1)
    circle_properties.update(circle_kwargs)

    line_properties = deepcopy(circle_properties)
    line_properties.pop('facecolor')

    fading_properties = dict(max_lw=2, min_lw=0.3)
    fading_properties.update(fading_kwargs)

    positions, angles = func(0)
    positions = np.asarray(positions, dtype=float)
    if angles is not None:
        angles = np.asarray(angles, dtype=float)
    Nparticles = positions.shape[0]

    if ax is None:
        ax = plt.gca()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    color_cycle = cycle(colors)

    particles = []
    dots = []
    lines = []
    trails = []
    text = {}

    if patches is not None:
        if not isinstance(patches.patches_type, Iterable):
            patches_type = [patches.patches_type]*Nparticles
        else:
            patches_type = patches.patches_type

        if not isinstance(patches.args[0], Iterable):
            patches_args = [patches.args for i in range(Nparticles)]
        else:
            patches_args = [[arg[i] for arg in patches.args] for i in range(Nparticles)]

    inv = ax.transData.transform
    for i in range(Nparticles): 
        pos = positions[i]
        color = next(color_cycle)

        if patches is not None:
            particles.append(patches_type[i](pos, *patches_args[i], edgecolor=color, animated=True, **circle_properties))
            ax.add_patch(particles[-1])

            if patches_type[i] is plt.Circle and angles is not None:
                radius = patches_args[i][0]
                lines.append(plt.Line2D([pos[0]-radius, pos[0]+radius], [pos[1], pos[1]], lw=circle_properties['linewidth'], color=color, animated=True, **line_properties))
                ax.add_line(lines[-1])
        else:
            dots.append(plt.Circle(inv(pos[:2]), 3, color=color, animated=True, transform=None))
            ax.add_patch(dots[-1])

        if trail > 0:
            if trail_type == 'normal':
                trails.append(ax.plot([pos[0]], [pos[1]], color=color, **trail_properties)[0])
            elif trail_type == 'fading':
                c = np.zeros((trail,4))
                c[:,0:3] = mpl.colors.to_rgb(color)
                c[:,3] = np.linspace(1,0,trail)
                lw = np.linspace(fading_properties['max_lw'],fading_properties['min_lw'],trail)
                trails.append(colored_plot([pos[0]], [pos[1]], c, ax=ax, linewidth=lw, **trail_properties))

        
        if number_labels:
            label = str(i+1)
            text[label] = ax.text(*pos, label, animated=True, **label_properties)
            text[label].set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                       path_effects.Normal()])

    ax.set_aspect('equal')

    if time is not None:
        text['clock'] = ax.text(.98,0.02, r"{0:.2f} {1}".format(0.0, time_unit), transform=ax.transAxes, horizontalalignment='right', animated=True, **time_properties)

    # history = np.zeros((trail,) + positions.shape, dtype=float)
    history = np.tile(positions, (trail,1,1))
    def update(t):
        nonlocal history

        positions, angles = func(t)
        positions = np.asarray(positions, dtype=float)
        if angles is not None:
            angles = np.asarray(angles, dtype=float)

        history = np.roll(history, 1, axis=0)
        if trail > 0:
            history[0] = positions

        for i in range(Nparticles): 
            pos = positions[i]
            if particles:
                particles[i].center = pos
                if angles is not None:
                    tran = mpl.transforms.Affine2D().rotate_around(*pos[:2], angles[i])
                    particles[i].set_transform(tran + ax.transData)

                if angles is not None and patches_type[i] is plt.Circle:
                    radius = patches_args[i][0]
                    lines[i].set_data([pos[0]-radius, pos[0]+radius], [pos[1], pos[1]])
                    lines[i].set_transform(rotation_transform(pos[:2], angles[i], ax=ax))

            if dots:
                dots[i].center = inv(pos[:2])

            if time is not None:
                text['clock'].set_text(r"{0:.2f} {1}".format(time[t], time_unit))

            if trail > 0:
                trails[i].set_data(history[:,i,0], history[:,i,1])
            
            if number_labels:
                text[str(i+1)].set_position(pos)
                
        
        return  trails + particles + lines + list(text.values()) + dots

    def init():
        positions, angles = func(0)
        positions = np.asarray(positions, dtype=float)
        angles = np.asarray(angles, dtype=float)

        history[...] = np.tile(positions, (trail,1,1))
        return  trails + particles + lines + list(text.values()) + dots

    anim = animation.FuncAnimation(ax.figure, update, frames=frames, init_func=init, blit=True, repeat=True, interval=30, **kwargs)
    return anim

def trajectory_animation(trajectory, patches=None, *args, **kwargs):
    """
    Create a 2D animation give trajectory data

    Arguments:
        trajectory     trajectory data for T steps, N particles
        patches        optional patches object to represent particles
    """
    if not isinstance(trajectory, stoked.trajectory):
        trajectory = stoked.trajectory(trajectory)

    if trajectory.orientation is not None:
        angles = stoked.quaternion_to_angles(trajectory.orientation)
    else:
        angles = [None]*len(trajectory.position)

    def func(i):
        return trajectory.position[i], angles[i]

    xlim = np.array([np.min(trajectory.position[...,0]),
                     np.max(trajectory.position[...,0])])
    ylim = np.array([np.min(trajectory.position[...,1]),
                     np.max(trajectory.position[...,1])])

    xbuff = np.abs(xlim[1] - xlim[0])*0.1
    xlim += (-xbuff, xbuff)
    ybuff = np.abs(ylim[1] - ylim[0])*0.1
    ylim += (-ybuff, ybuff)

    return animation_2d(func, patches=patches, frames=len(trajectory.position), xlim=xlim, ylim=ylim, *args, **kwargs)
