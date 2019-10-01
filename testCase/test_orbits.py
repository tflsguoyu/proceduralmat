import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def f_euler(A, u, N):
    orbit = np.zeros((N,2))

    dt = 2*np.pi/N
    for i in range(N):
        u = u + dt * A @ u
        orbit[i] = u
    return orbit

def trapezoidal(A, u, N):
    p = len(u)
    orbit = np.zeros((N,p))

    dt = 2*np.pi/N
    for i in range(N):
        u = la.inv(np.eye(p) - dt/2 * A) @ (np.eye(p) + dt/2 * A) @ u
        orbit[i] = u
    return orbit


def leapfrog(A, u, N):
    orbit = np.zeros((N,2))

    dt = 2*np.pi/N
    for i in range(N):
        u[1] = u[1] + dt/2 * A[1] @ u
        u[0] = u[0] + dt * A[0] @ u
        u[1] = u[1] + dt/2 * A[1] @ u
        orbit[i] = u
    return orbit

A = np.array([[0,1],[-1,0]])
u = np.array([1.0,0.0])
N = 32

orbit = f_euler(A, u, N)

plt.subplot(131)
plt.plot(orbit[:, 0], orbit[:,1], 'o')
plt.axis('square')

orbit2 = trapezoidal(A, u, N)

plt.subplot(132)
plt.plot(orbit2[:, 0], orbit2[:,1], 'o')
plt.axis('square')

orbit3 = leapfrog(A, u, N)

plt.subplot(133)
plt.plot(orbit3[:, 0], orbit3[:,1], 'o')
plt.axis('square')

plt.show()