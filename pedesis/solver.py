import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as kb
from tqdm import tqdm
from functools import partial

def no_force(t, rvec):
    return np.zeros_like(rvec)

def uniform_force(t, rvec, Fx):
    F = np.zeros_like(rvec)
    F[:,0] = Fx
    return F

def harmonic_force(t, rvec, k=1):
    return -k*rvec

def gaussian_force(t, rvec, A=1, w=1):
    r = np.linalg.norm(rvec, axis=1)
    return -A*rvec/w**2*np.exp(-r**2/(2*w**2))

def bigaussian_force(t, rvec, rvec1, rvec2, A1=1, w1=1, A2=1, w2=1):
    r1 = np.linalg.norm(rvec - rvec1, axis=1)[:,np.newaxis]
    r2 = np.linalg.norm(rvec - rvec2, axis=1)[:,np.newaxis]
    F1 = -A1*(rvec - rvec1)/w1**2*np.exp(-r1**2/(2*w1**2))
    F2 = -A2*(rvec - rvec2)/w2**2*np.exp(-r2**2/(2*w2**2))

    return F1 + F2

class brownian_dynamics:
    def __init__(self, position, Fext, damping, temperature, dt):
        """
        Arguments:
            position[N,3]    initial position of N particles
            Fext(t, r)       external force given time t, position r[N,3] and returns force F[N,3]
            damping          damping coefficient (F = -(damping)*(velocity))
            temperature      system temperature
            dt               time-step
        """
        self.position = np.asarray(position)
        self.Fext = Fext
        self.damping = damping
        self.temperature = temperature
        self.dt = dt
        self.time = 0

        self.alpha = 1/self.damping
        self.beta = np.sqrt(2*kb*self.temperature/(dt*self.damping))

        self.velocity = np.zeros_like(position)

    def step(self):
        """
        Time-step the positions by dt
        """
        F = self.Fext(self.time, self.position)
        noise = np.random.normal(size=self.position.shape) 
        v1 = (self.alpha*F + self.beta*noise)
        r_predict = self.position + self.dt*v1
        self.time += self.dt

        F_predict = self.Fext(self.time, r_predict)
        v2 = (self.alpha*F_predict + self.beta*noise)
        self.velocity = 0.5*(v1 + v2)
        self.position = self.position + 0.5*self.dt*(v1 + v2)


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

def uniform_example():
    position = [[0,0,0]]
    damping = 1
    temperature = .1
    dt = 10
    Fx = 10e-13

    Nsteps = 50000

    history = np.zeros([Nsteps,3], dtype=float)
    velocity = np.zeros([Nsteps,3], dtype=float)
    sim = brownian_dynamics(position, partial(uniform_force, Fx=Fx), damping, temperature, dt)

    for i in tqdm(range(Nsteps)):
        history[i] = sim.position.squeeze()
        velocity[i] = sim.velocity.squeeze()
        sim.step()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(history[:,0], history[:,1], lw=.5)

    fig, ax = plt.subplots()
    ax.hist(velocity[:,0], bins=100)

    v_drift_theory = Fx/(damping)
    v_drift_sim = np.average(velocity[:,0])
    print(v_drift_theory, v_drift_sim)

    plt.show()


# delta = 1e-11
# x = np.linspace(-3*delta, 3*delta, 1000)
# x1 = -delta
# x2 = delta
# A1 = A2 = 1e-8
# w1 = w2 = delta/1.4
# U = -A1*np.exp(-(x - x1)**2/(2*w1**2)) - A2*np.exp(-(x - x2)**2/(2*w2**2))
# plt.plot(x, U)
# plt.show()
harmonic_example()
