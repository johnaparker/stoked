from pedesis import brownian_dynamics
import matplotlib.pyplot as plt
from tqdm import tqdm

def no_force(t, rvec):
    return np.zeros_like(rvec)

def harmonic_example():
    position = [[0,0,0]]
    damping = 1
    temperature = .5
    dt = 10

    Nsteps = 100000
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    history = np.zeros([Nsteps,3], dtype=float)

    sim = brownian_dynamics(position, damping, temperature, dt, force=no_force)

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        sim.step()

    ax.plot(history[:,0], history[:,1], lw=.5)
    ax.legend()

    plt.show()
