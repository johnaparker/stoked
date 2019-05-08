"""
Utility functions
"""
import numpy as np
import quaternion
from scipy.integrate import cumtrapz

def quaternion_to_angles(quat, reference=None):
    """
    Convert a quaternion array to an angle representation

    Arguments:
        quat        [T,...] quaternion trajectory of T time-steps
        reference   reference frame (as quaternion) around which to compute angles (default: z-axis)
    """
    if reference is not None:
        quat = np.invert(reference)*quat

    ### calculates differntial angle at each time-step, and cumtrapz to obtain angle
    quat_s = np.roll(quat, 1, axis=0)
    Q = quat*np.invert(quat_s)
    axis_angle = quaternion.as_rotation_vector(Q)
    d_angle = axis_angle[...,2]
    d_angle[0] = 0    # first entry is unphysical, so set to 0

    ### obtain the initial angles; multiply phi by 2 if theta = 0 for proper conversion
    theta, phi = np.moveaxis(quaternion.as_spherical_coords(quat[0]), -1, 0)
    idx = (theta == 0)
    phi[idx] *= 2

    angle = phi + cumtrapz(d_angle, axis=0, initial=0)
    return angle
