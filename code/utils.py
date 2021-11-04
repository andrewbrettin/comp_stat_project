import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

class KalmanFilter():
    def __init__(self, dt, x, u):
        # Timestep
        self.dt = dt

        # Propagator matrix
        self.A = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Propagator
        self.B = np.array([
            [0.5*self.dt**2, 0, 0],
            [0, 0.5*self.dt**2, 0],
            [0, 0, 0.5*self.dt**2],
            [self.dt, 0, 0],
            [0, self.dt, 0],
            [0, 0, self.dt]
        ])

        self.H = np.concatenate([np.eye(3), np.zeros((3,3))], axis=1)

        self.m = 3
        self.n = 6

        self.R = np.eye(self.m)
        self.Q = np.eye(self.n)

        self.P = np.eye(self.n)
        self.P_prior = np.empty((self.n, self.n))

        self.x = np.concatenate([x, np.zeros(3)])    # Assume 0 initial velocity
        self.u = u
        self.x_prior = np.empty_like(self.x)

    def forecast(self):
        # 
        self.x_prior = self.A @ self.x + self.B @ self.u
        self.P_prior = self.A @ self.P @ np.transpose(self.A) + self.Q
        return self.x[0:3]

    def update(self, z):
        K = (
            self.P_prior @ np.transpose(self.H) 
            @ inv(self.H @ self.P_prior @ np.transpose(self.H) + self.R)
        )
        self.x = self.x_prior + K @ (z - self.H @ self.x_prior)
        self.P = (np.eye(self.n) - K @ self.H) @ self.P_prior
        return self.x[0:3]