"""Utility functions."""

import numpy as np
import quaternion
from scipy.integrate import cumulative_trapezoid


def quaternion_to_angles(quat, reference=None):
    """Convert a quaternion array to an angle representation.

    Arguments:
        quat        [T,...] quaternion trajectory of T time-steps
        reference   reference frame (as quaternion) around which to compute angles (default: z-axis)
    """
    if reference is not None:
        quat = np.invert(reference) * quat

    ### calculates differntial angle at each time-step, and cumtrapz to obtain angle
    quat_s = np.roll(quat, 1, axis=0)
    Q = quat * np.invert(quat_s)
    axis_angle = quaternion.as_rotation_vector(Q)
    d_angle = axis_angle[..., 2]
    d_angle[0] = 0  # first entry is unphysical, so set to 0

    ### obtain the initial angles; multiply phi by 2 if theta = 0 for proper conversion
    theta, phi = np.moveaxis(quaternion.as_spherical_coords(quat[0]), -1, 0)
    idx = theta == 0
    phi[idx] *= 2

    angle = phi + cumulative_trapezoid(d_angle, axis=0, initial=0)
    return angle


def hexagonal_lattice_layers(L):
    """Return a hexagonal lattice with unit spacing and L layers.

    Creates a 2D hexagonal lattice in the xy-plane (z=0) with the first particle
    at the origin and subsequent layers forming concentric hexagonal rings.

    Arguments:
        L: Number of hexagonal layers around the central particle

    Returns:
        Array of shape (N, 3) containing particle positions in 3D space
    """
    k1 = np.array([1, 0, 0], dtype=float)
    k2 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3), 0], dtype=float)

    lattice = [np.zeros(3)]

    for step in range(1, L + 1):
        lattice.append(step * k1)

        for direc in [k2 - k1, -k1, -k2, -k2 + k1, k1, k2]:
            for _ in range(step):
                lattice.append(lattice[-1] + direc)
        lattice.pop()

    return np.asarray(lattice, dtype=float)


def hexagonal_lattice_particles(N):
    """Return a hexagonal lattice with unit spacing and N particles.

    Creates a 2D hexagonal lattice in the xy-plane (z=0) with N particles.
    The lattice is centered at the origin with the first particle at (0, 0, 0).

    Arguments:
        N: Number of particles to include in the lattice

    Returns:
        Array of shape (N, 3) containing particle positions in 3D space
    """
    L = int(np.ceil(np.sqrt((N - 1) / 3)))
    lattice = hexagonal_lattice_layers(L)
    return lattice[:N]
