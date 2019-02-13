import numpy as np

def hard_sphere_collisions(radii, kn):
    def force(t, rvec):
        F = np.zeros_like(rvec)
        Nparticles = len(rvec)

        if np.isscalar(radii):
            rad = np.full(Nparticles, radii, dtype=float)
        else:
            rad = radii

        for i in range(0,Nparticles):
            for j in range(i+1,Nparticles):
                d = rvec[i] - rvec[j]
                r = np.linalg.norm(d)
                if r < rad[i] + rad[j]:
                    overlap = abs(r - rad[i] + rad[j])

                    r_hat = d/r
                    Fn = r_hat*kn*overlap**1.5

                    F[i] += Fn
                    F[j] -= Fn
        return F

    return force
