import numpy as np

class CVKalmanFilter:
    def __init__(self, dt, q, r):
        self.model_type = "CV"
        self.dt = dt

        # State vector [x, y, vx, vy]
        self.x = np.zeros(4)

        # State covariance
        self.P = np.eye(4) * 500.0

        # Transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])

        # Process noise covariance
        q4 = q * dt**4 / 4.0    
        q3 = q * dt**3 / 2.0  
        q2 = q * dt**2          

        self.Q = np.array([
            [q4, 0,  q3, 0 ],
            [0,  q4, 0,  q3],
            [q3, 0,  q2, 0 ],
            [0,  q3, 0,  q2 ]
])
        # Measurement matrix (we observe x,y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise
        self.R = np.eye(2) * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(-1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return self.x.copy()
