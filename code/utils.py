import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

class KalmanFilter():
    def __init__(self, dt, x, R=np.eye(3), Q=np.eye(6)):
        self.m = 3
        self.n = 6
        
        # Timestep
        self.dt = dt

        # Propagator matrix
        self.A = np.eye(6)
        self.A[0,1] = dt
        self.A[2,3] = dt
        self.A[4,5] = dt

        # Observation matrix
        self.H = np.zeros((3,6))
        self.H[0,0] = 1
        self.H[1,2] = 1
        self.H[2,4] = 1
        
        # Propagation covariance
        self.Q = Q
        
        # Observation covariance
        self.R = R

        # Intialize covariance matrix
        self.P = np.eye(self.n)
        self.P_prior = np.empty((self.n, self.n))

        self.x = np.array([x[0], 0, x[1], 0, x[2], 0])    # Assume 0 initial velocity
        self.x_prior = np.empty_like(self.x)

    def set_Q(self, *args):
        std_1, std_2, std_3 = args
        
        block = np.array([
            [0.25 * self.dt**4, 0.5 * self.dt**3],
            [0.5 * self.dt**3, self.dt**2]
        ])
        zeros = np.zeros((2,2))
        
        self.Q = np.block([
            [std_1*block, zeros, zeros],
            [zeros, std_2*block, zeros],
            [zeros, zeros, std_3*block]
        ])

    def forecast(self):
        self.x_prior = self.A @ self.x
        self.P_prior = self.A @ self.P @ np.transpose(self.A) + self.Q
        return self.x_prior[::2]

    def update(self, z):
        K = (
            self.P_prior @ np.transpose(self.H) 
            @ inv(self.H @ self.P_prior @ np.transpose(self.H) + self.R)
        )
        self.x = self.x_prior + K @ (z - self.H @ self.x_prior)
        self.P = (np.eye(self.n) - K @ self.H) @ self.P_prior
        return self.x[::2]