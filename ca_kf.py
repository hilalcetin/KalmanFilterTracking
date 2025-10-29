import numpy as np

class CAKalmanFilter:
    def __init__(self, dt, q, r):
        self.model_type = "CA"
        self.dim_x = 6  # [x, y, vx, vy, ax, ay]
        self.dim_z = 2

        self.x = np.zeros(self.dim_x)
        self.P = np.eye(self.dim_x) * 500.

        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])

        self.R = np.eye(self.dim_z) * r
        self.Q = np.eye(self.dim_x) * q

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(-1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return self.x.copy()
