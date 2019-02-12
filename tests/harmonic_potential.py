from pedesis import brownian_dynamics
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

def harmonic_force(t, rvec, k=1):
    return -k*rvec

def harmonic_example():
    position = [[0,0,0]]
    damping = 1
    temperature = .5
    dt = 10

    Nsteps = 100000
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    for Fext in [no_force, harmonic_force]:
        history = np.zeros([Nsteps,3], dtype=float)

        if Fext is no_force:
            sim = brownian_dynamics(position, Fext, damping, temperature, dt)
        else:
            sim = brownian_dynamics(position, partial(Fext, k=.01), damping, temperature, dt)

        for i in tqdm(range(Nsteps)):
            history[i] = sim.position.squeeze()
            sim.step()

        ax.plot(history[:,0], history[:,1], lw=.5, label=Fext.__name__)
        ax.legend()

        if Fext is harmonic_force:
            fig, ax = plt.subplots()
            ax.hexbin(history[:,0], history[:,1])
            ax.set_aspect('equal')

            fig, ax = plt.subplots()
            rad = np.linalg.norm(history, axis=1)
            hist, edges = np.histogram(rad, bins=100)

            rad = edges[1:]
            counts = hist/(4*np.pi*rad**2)
            E = -kb*temperature*np.log(counts)
            E -= E[6]
            ax.plot(rad, E)
            ax.plot(rad, 0.5*.01*rad**2, 'o', ms=3)



    plt.show()


