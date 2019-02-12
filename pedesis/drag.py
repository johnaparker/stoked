import numpy as np

def sphere_translation_drag(radius, viscosity):
    """translational drag coeffecient for a sphere with given radius and surrounding viscosity"""
    return 6*np.pi*radius*viscosity

def sphere_rotation_drag(radius, viscosity):
    """rotational drag coeffecient for a sphere with given radius and surrounding viscosity"""
    return 8*np.pi*radius**3*viscosity

def ellipsoid_translation_drag(radii, viscosity):
    """translational drag coeffecient for an ellipsoid with given radii and surrounding viscosity"""
    pass

def ellipsoid_rotation_drag(radii, viscosity):
    """rotational drag coeffecient for an ellipsoid with given radii and surrounding viscosity"""
    pass
