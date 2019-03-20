import numpy as np
from pedesis import interactions

class van_der_waals_spheres(interactions):
    def __init__(self, radius):
        self.radius = radius

class van_der_waals_interface(interactions):
    pass
