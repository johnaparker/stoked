from pedesis import brownian_dynamics
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm

def gaussian_force(t, rvec, A=1, w=1):
    r = np.linalg.norm(rvec, axis=1)
    return -A*rvec/w**2*np.exp(-r**2/(2*w**2))

def bigaussian_force(t, rvec, rvec1, rvec2, A1=1, w1=1, A2=1, w2=1):
    r1 = np.linalg.norm(rvec - rvec1, axis=1)[:,np.newaxis]
    r2 = np.linalg.norm(rvec - rvec2, axis=1)[:,np.newaxis]
    F1 = -A1*(rvec - rvec1)/w1**2*np.exp(-r1**2/(2*w1**2))
    F2 = -A2*(rvec - rvec2)/w2**2*np.exp(-r2**2/(2*w2**2))

    return F1 + F2


def gaussian_example():
    position = [[0,0,0]]
    damping = 1
    temperature = .1
    dt = 1

    Nsteps = 10000
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    history = np.zeros([Nsteps,3], dtype=float)
    sim = brownian_dynamics(position, partial(gaussian_force), damping, temperature, dt)

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        sim.step()

    ax.plot(history[:,0], history[:,1], lw=.5)

    plt.show()

def bigaussian_example():
    delta = 1e-10
    r1 = [-delta*.9, 0, 0]
    r2 = [delta*.9, 0, 0]
    w1 = w2 = delta/1.3
    A1 = A2 = 1e-22
    position = [[-delta,0,0]]
    damping = 1
    temperature = .4
    dt = 40

    Nsteps = 1000000
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    history = np.zeros([Nsteps,3], dtype=float)
    sim = brownian_dynamics(position, partial(bigaussian_force, rvec1=r1, rvec2=r2, A1=A1, A2=A2, w1=w1, w2=w2), damping, temperature, dt)
    # sim = brownian_dynamics(position, no_force, damping, temperature, dt)

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        sim.step()

    ax.plot(history[:,0], history[:,1], lw=.5)

    fig, ax = plt.subplots()
    ax.hexbin(history[:,0], history[:,1], extent=[-1.5*delta, 1.5*delta, -delta, delta])
    ax.set_aspect('equal')
    plt.show()
