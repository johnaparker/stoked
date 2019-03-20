import numpy as np
from pedesis import interactions

class hard_sphere_collisions(interactions):
    def __init__(self, radii, kn):
        # self.radii = pedesis.array(radii)
        self.radii = radii
        self.kn = kn

    def force(self):
        Nparticles = len(self.position)
        if np.isscalar(self.radii):
            rad = np.full(Nparticles, self.radii, dtype=float)
        else:
            rad = np.asarray(self.radii, dtype=float)

        F = np.zeros_like(self.position)

        for i in range(0,Nparticles):
            for j in range(i+1,Nparticles):
                d = self.position[i] - self.position[j]
                r = np.linalg.norm(d)
                if r < rad[i] + rad[j]:
                    overlap = abs(r - rad[i] + rad[j])

                    r_hat = d/r
                    Fn = r_hat*self.kn*overlap**1.5

                    F[i] += Fn
                    F[j] -= Fn
        return F

    def torque(self):
        return np.zeros_like(self.position)

class hard_sphere_plane_collision(interactions):
    def __init__(self, radii, kn):
        self.radii = radii
        self.kn = kn
