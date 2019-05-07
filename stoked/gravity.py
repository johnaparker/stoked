from scipy import constants
import numpy as np

density_water = 997

def gravity(volume, density, density_medium=density_water, direction=[0,0,-1]):
    direction = np.asarray(direction, dtype=float)

    def force(time, position, orientation):
        if np.isscalar(volume):
            vol = np.full(len(position), volume)
        else:
            vol = np.asarray(volume, dtype=float)

        if np.isscalar(density):
            den = np.full(len(position), density)
        else:
            den = np.asarray(density, dtype=float)

        mass = vol*(den - density_medium)
        F = constants.g*mass
        return np.outer(F, direction)

    return force

def gravity_sphere(radius, density, density_medium=density_water, direction=[0,0,-1]):
    volume = 4/3*np.pi*radius**3
    return gravity(volume, density, density_medium, direction)

def gravity_ellipsoid(radii, density, density_medium=density_water, direction=[0,0,-1]):
    volume = 4/3*np.pi*np.product(radii, axis=-1)
    return gravity(volume, density, density_medium, direction)
