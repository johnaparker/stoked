from itertools import cycle
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from .colored_plot import colored_plot
import matplotlib.patheffects as path_effects
from tqdm import tqdm

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
def trajectory_animation(coordinates, radii, projection, angles=None, colors=['C0'], ax=None,
        xlim=None, ylim=None, time=None, time_unit='T', number_labels=False, trail=0, trail_type='normal',
        time_kwargs={}, label_kwargs={}, circle_kwargs={}, trail_kwargs={}, fading_kwargs={}, **kwargs):
    """create a 2D animation of trajectories

            coordinates[T,N,3]         particle x,y,z coordinates 
            radii[N] or scalar         particle radii
            projection ('x','y','z')   which plane to project 3D trajectories onto
            angles[T,N]                particle angles
            colors                     list of colors to cycle through
            ax (default None)          specify the axes of the animation
            xlim[2]                    min,max values of x-axis
            ylim[2]                    min,max values of y-axis
            time[N]                    display the time (in time_units)
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

    coordinates = np.asarray(coordinates)
    radii = atleast(radii, dim=1, length=coordinates.shape[1])
    if angles is not None: 
        angles = np.asarray(angles)

    idx = [0,1,2]
    idx.pop(ord(projection) - ord('x'))
    coordinates = coordinates[...,idx]
    Nt,Nparticles,_ = coordinates.shape

    if ax is None:
        ax = plt.gca()
    if xlim is None:
        xlim = np.array([np.min(coordinates[...,0] - 1.3*radii),
                         np.max(coordinates[...,0] + 1.3*radii)])
    if ylim is None:
        ylim = np.array([np.min(coordinates[...,1] - 1.3*radii),
                         np.max(coordinates[...,1] + 1.3*radii)])
    color_cycle = cycle(colors)

    circles = []
    lines = []
    trails = []
    text = {}

    for i in range(Nparticles): 
        coordinate = coordinates[0,i]
        color = next(color_cycle)

        circles.append(plt.Circle(coordinate, radii[i], edgecolor=color, animated=True, **circle_properties))
        # circles.append(mpl.patches.Ellipse(coordinate, 2*radii[i], 3*radii[i], edgecolor=color, animated=True, **circle_properties))
        ax.add_artist(circles[-1])

        if angles is not None:
            # lines.append(plt.Line2D([coordinate[0]-radii[i], coordinate[0]+radii[i]], [coordinate[1], coordinate[1]], lw=circle_properties['linewidth'], color=color, animated=True, zorder=circle_properties['zorder']))
            lines.append(plt.Line2D([coordinate[0]-radii[i], coordinate[0]+radii[i]], [coordinate[1], coordinate[1]], color=color, animated=True, **line_properties))
            ax.add_line(lines[-1])

        if trail > 0:
            if trail_type == 'normal':
                trails.append(ax.plot([coordinate[0]], [coordinate[1]], color=color, **trail_properties)[0])
            elif trail_type == 'fading':
                c = np.zeros((trail,4))
                c[:,0:3] = mpl.colors.to_rgb(color)
                c[:,3] = np.linspace(1,0,trail)
                lw = np.linspace(fading_properties['max_lw'],fading_properties['min_lw'],trail)
                trails.append(colored_plot([coordinate[0]], [coordinate[1]], c, ax=ax, linewidth=lw, **trail_properties))

        
        if number_labels:
            label = str(i+1)
            text[label] = ax.text(*coordinate, label, animated=True, **label_properties)
            text[label].set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                       path_effects.Normal()])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    if time is not None:
        text['clock'] = ax.text(.98,0.02, r"{0:.2f} {1}".format(0.0, time_unit), transform=ax.transAxes, horizontalalignment='right', animated=True, **time_properties)

    def update(t):
        for i in range(Nparticles): 
            coordinate = coordinates[t,i]
            circles[i].center = coordinate
            # tran = mpl.transforms.Affine2D().rotate_around(*coordinate, 1.0*t/100)
            # circles[i].set_transform(tran + ax.transData)

            if angles is not None:
                lines[i].set_data([coordinate[0]-radii[i], coordinate[0]+radii[i]], [coordinate[1], coordinate[1]])
                lines[i].set_transform(rotation_transform(coordinate, angles[t,i], ax=ax))

            if time is not None:
                text['clock'].set_text(r"{0:.2f} {1}".format(time[t], time_unit))

            if trail > 0:
                tmin = max(0,t-trail)
                trails[i].set_data(coordinates[t:tmin:-1,i,0], coordinates[t:tmin:-1,i,1])
            
            if number_labels:
                text[str(i+1)].set_position(coordinate + np.array([-radii[i], radii[i]]))
                # text[str(i+1)].set_position(coordinate)
                
        
        return  trails + circles + lines + list(text.values())

    anim = animation.FuncAnimation(ax.figure, update, frames=np.arange(0,Nt,1), blit=True, repeat=True, **kwargs)
    return anim
