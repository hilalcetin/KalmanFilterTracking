import numpy as np

class CTKalmanFilter:
    """Coordinated Turn (CT) Kalman Filter: state = [x, y, vx, vy, omega]"""
    def __init__(self, dt, q_pos=None, q_vel=None, q_omega=None, r=None):
        self.model_type = "CT"
        self.dt = dt

        # State vector [x, y, vx, vy, omega]
        self.x = np.zeros(5)

        # State covariance
        self.P = np.eye(5) * 500.0

        # Measurement matrix (observe x, y)
        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0]])

        # Measurement noise
        self.R = np.eye(2) * (1.0 if r is None else r)

        # Process noise (separate for position, velocity, omega)
        q_pos = 0.1 if q_pos is None else q_pos
        q_vel = 0.1 if q_vel is None else q_vel
        q_omega = 0.01 if q_omega is None else q_omega
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_omega])

    def predict(self):
        x, y, vx, vy, omega = self.x
        dt = self.dt

        # --- State propagation ---
        if abs(omega) < 1e-5:
            # Straight motion
            x_new = x + vx * dt
            y_new = y + vy * dt
            vx_new = vx
            vy_new = vy
        else:
            # Coordinated turn motion
            x_new = x + (vx * np.sin(omega*dt) - vy * (1 - np.cos(omega*dt))) / omega
            y_new = y + (vy * np.sin(omega*dt) + vx * (1 - np.cos(omega*dt))) / omega
            vx_new = vx * np.cos(omega*dt) - vy * np.sin(omega*dt)
            vy_new = vx * np.sin(omega*dt) + vy * np.cos(omega*dt)

        self.x = np.array([x_new, y_new, vx_new, vy_new, omega])

        # --- Jacobian F matrix ---
        F = np.eye(5)
        if abs(omega) < 1e-5:
            F[0,2] = dt
            F[1,3] = dt
        else:
            sin_ωdt = np.sin(omega*dt)
            cos_ωdt = np.cos(omega*dt)

            # ∂x/∂vx, ∂x/∂vy, ∂x/∂ω
            F[0,2] = sin_ωdt / omega
            F[0,3] = -(1 - cos_ωdt) / omega
            F[0,4] = (dt*(vx*cos_ωdt - vy*sin_ωdt) - (vx*sin_ωdt - vy*(1 - cos_ωdt))/omega) / omega

            # ∂y/∂vx, ∂y/∂vy, ∂y/∂ω
            F[1,2] = (1 - cos_ωdt) / omega
            F[1,3] = sin_ωdt / omega
            F[1,4] = (dt*(vx*sin_ωdt + vy*cos_ωdt) - (vy*sin_ωdt + vx*(1 - cos_ωdt))/omega) / omega

            # ∂vx/∂vx, ∂vx/∂vy, ∂vx/∂ω
            F[2,2] = cos_ωdt
            F[2,3] = -sin_ωdt
            F[2,4] = -vx*dt*sin_ωdt - vy*dt*cos_ωdt

            # ∂vy/∂vx, ∂vy/∂vy, ∂vy/∂ω
            F[3,2] = sin_ωdt
            F[3,3] = cos_ωdt
            F[3,4] = vx*dt*cos_ωdt - vy*dt*sin_ωdt

            # ∂ω/∂ω = 1 already

        # --- Covariance update ---
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        z = np.asarray(z).reshape(-1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P

    def step(self, z):
        self.predict()
        self.update(z)
        return self.x.copy()