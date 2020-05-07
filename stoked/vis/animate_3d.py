import numpy as np
import matplotlib as mpl
from itertools import cycle
import quaternion
import stoked
from collections.abc import Iterable
import matplotlib.colors as mcolors
from stoked.vis._internal import patches

def sphere_patches(radius):
    import vpython
    return patches(vpython.sphere, dict(radius=radius))

def ellipsoid_patches(rx, ry, rz):
    import vpython
    return patches(vpython.ellipsoid, dict(length=2*rx, height=2*ry, width=2*rz))

def trajectory_animation_3d(trajectory, patches, wait=False, repeat=True, colors=None, opacity=1, trail=None, axes=False, axes_length=1, grid=False):
    """Create a 3D trajectory animation with VPython
        
        Arguments:
            trajectory        trajectory data for T steps, N particles
            patches           patches object to represent particle geometry
            colors            list of colors to cycle through
            trail             length of particle trail (default: no trail)
            axes              include x,y,z axes for each particle (default: False)
            axes_length       length of axes, if set (default: 1)
    """
    import vpython
    vec = vpython.vector

    if not isinstance(trajectory, stoked.trajectory):
        trajectory = stoked.trajectory(trajectory)
    coordinates = trajectory.position

    Nsteps = coordinates.shape[0]
    Nparticles = coordinates.shape[1]

    if trajectory.orientation is not None:
        orientations = trajectory.orientation
    else:
        orientations = np.full((Nsteps,Nparticles), quaternion.one)

    if not isinstance(patches.patches_type, Iterable):
        patches_type = [patches.patches_type]*Nparticles
    else:
        patches_type = patches.patches_type

    patches_args = [dict() for i in range(Nparticles)]
    for key, value in patches.args.items():
        if not isinstance(value, Iterable):
            for i in range(Nparticles):
                patches_args[i][key] = value
        else:
            for i in range(Nparticles):
                patches_args[i][key] = value[i]

    if colors is None:
        colors = mcolors.TABLEAU_COLORS
    elif isinstance(colors, str):
        colors = [colors]
    elif not isinstance(colors, Iterable):
        colors = [colors]
    color_cycle = cycle(colors)

    make_trail = False if trail is None else True

    scene = vpython.canvas(background=vec(1,1,1))
    objs = []
    arrows = []
    trails = []

    for i in range(Nparticles):
        color = vec(*mpl.colors.to_rgb(next(color_cycle)))
        pos = vec(*coordinates[0,i])
        particle = patches_type[i](pos=pos, color=color,
                   opacity=opacity, **patches_args[i])
        objs.append(particle)

        if make_trail:
            def my_center(num):
                def center():
                    return objs[num].pos + objs[num].axis/2
                return center
            trails.append(vpython.attach_trail(my_center(i), color=color, retain=trail))

        if axes:
            arrow_x = vpython.arrow(pos=pos, axis=vec(1,0,0), scale=axes_length, color=vec(0,0,0),
                          shaftwidth=5)
            arrow_y = vpython.arrow(pos=pos, axis=vec(0,1,0), scale=axes_length, color=vec(0,0,0),
                          shaftwidth=5)
            arrow_z = vpython.arrow(pos=pos, axis=vec(0,0,1), scale=axes_length, color=vpython.color.red,
                          shaftwidth=5)

            arrows.append([arrow_x, arrow_y, arrow_z])


    for i,obj in enumerate(objs):
        rot = quaternion.as_rotation_matrix(orientations[0,i])
        a = obj.axis
        b = vec(*rot[:,2])
        a /= vpython.mag(a)
        b /= vpython.mag(b)
        axis = vpython.cross(a,b)
        angle = vpython.acos(vpython.dot(a,b))
        obj.rotate(angle=angle, axis=axis, origin=obj.pos + obj.axis/2)
        obj.pos -= obj.axis/2
        if axes:
            for j,arrow in enumerate(arrows[i]):
                arrow.pos = vec(*coordinates[0,i])
                arrow.axis = vec(*rot[:,j])*arrow.scale

    if grid:
        vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(300,0,0), shaftwidth=2, color=vpython.color.black)
        vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(0,300,0), shaftwidth=2, color=vpython.color.black)
        vpython.arrow(pos=vpython.vector(0,0,0), axis=vpython.vector(0,0,300), shaftwidth=2, color=vpython.color.black)

    if repeat:
        trange = cycle(range(1,coordinates.shape[0]))
    else:
        trange = range(1,coordinates.shape[0])

    for t in trange:
        if t == 1:
            if wait:
                scene.waitfor('click')
            for trail in trails:
                trail.clear()
        for i,obj in enumerate(objs):
            rot = quaternion.as_rotation_matrix(orientations[t,i])

            a = obj.axis
            b = vec(*rot[:,2])
            a /= vpython.mag(a)
            b /= vpython.mag(b)
            axis = vpython.cross(a,b)
            angle = vpython.acos(vpython.dot(a,b))
            obj.rotate(angle=angle, axis=axis, origin=obj.pos + obj.axis/2)
            
            if patches_type[i] in (vpython.cylinder, vpython.arrow, vpython.cone, vpython.pyramid):
                obj.pos = vec(*coordinates[t,i]) - obj.axis/2
            else:
                obj.pos = vec(*coordinates[t,i])
            if axes:
                for j,arrow in enumerate(arrows[i]):
                    arrow.pos = vec(*coordinates[t,i])
                    arrow.axis = vec(*rot[:,j])*arrow.scale

        vpython.rate(30)


if __name__ == '__main__':
    coordinates = np.zeros([100, 2, 3])
    coordinates[:,0,0] = np.linspace(10, 100, 100)
    coordinates[:,1,0] = -np.linspace(10, 100, 100)

    patches = sphere_patches([5,10])
    trajectory_animation_3d(coordinates, patches, opacity=.5, colors='C0', trail=100, wait=True)
