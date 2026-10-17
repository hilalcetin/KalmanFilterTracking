import numpy as np


def project_to_common(x, model_type):
    """Model state'ini geniş ortak uzaya projekte et: [x, y, vx, vy, ax, ay, ω]"""
    if model_type == "CV":
        return np.hstack([x, [0, 0, 0]])   # CV: ax=ay=0, ω=0
    elif model_type == "CA":
        return np.hstack([x[:6], [0]])     # CA: ω=0
    elif model_type == "CT":
        return np.hstack([x[:4], [0, 0], x[4:]])  # CT: ax=ay=0
    else:
        raise ValueError("Unknown model type")


def expand_from_common(x_common, model_type):
    """Geniş ortak state [x,y,vx,vy,ax,ay,ω] → modele indir."""
    if model_type == "CV":
        return x_common[:4]                  # sadece [x,y,vx,vy]
    elif model_type == "CA":
        return x_common[:6]                  # [x,y,vx,vy,ax,ay]
    elif model_type == "CT":
        return np.hstack([x_common[:4], x_common[6:]])  # [x,y,vx,vy,ω]
    else:
        raise ValueError("Unknown model type")


def project_cov_to_common(P, model_type):
    """Model kovaryansini genis ortak uzaya göm."""
    dim_common = 7  # [x,y,vx,vy,ax,ay,ω]
    P_common = np.zeros((dim_common, dim_common))

    if model_type == "CV":
        P_common[:4, :4] = P  # sadece ilk 4x4
    elif model_type == "CA":
        P_common[:6, :6] = P  # ilk 6x6
    elif model_type == "CT":
        # CT state: [x,y,vx,vy,ω] → embed: [:4] ve son eleman [6]
        P_common[:4, :4] = P[:4, :4]
        P_common[:4, 6] = P[:4, 4]
        P_common[6, :4] = P[4, :4]
        P_common[6, 6] = P[4, 4]
    else:
        raise ValueError("Unknown model type")

    return P_common


class IMM:
    def __init__(self, models, mu, PI):
        """
        models: [CV, CA, CT ...] filtre nesneleri
        mu: başlangıç model olasılıkları (numpy array)
        PI: geçiş olasılık matrisi
        """
        self.models = models
        self.mu = mu / np.sum(mu)  # normalize
        self.PI = PI
        self.N = len(models)

        # IMM state & covariance (ortak uzayda)
        self.x = np.zeros(7)
        self.P = np.eye(7)

        # yardımcı değişkenler
        self.likelihood = np.zeros(self.N)
        self.omega = np.zeros((self.N, self.N))
        self.cbar = np.zeros(self.N)

        # ilk state estimate
        self._compute_state_estimate()

    # -------------------
    # 1. Mixing Probabilities
    # -------------------
    def _compute_mixing_probabilities(self):
        self.cbar = self.mu @ self.PI  # (N,)
        for i in range(self.N):
            for j in range(self.N):
                if self.cbar[j] > 0:
                    self.omega[i, j] = (self.PI[i, j] * self.mu[i]) / self.cbar[j]
                else:
                    self.omega[i, j] = 0.0

    # -------------------
    # 2. State & Covariance Karışımı
    # -------------------
    def _mix_states(self):
        mixed_x = []
        mixed_P = []

        for j, model in enumerate(self.models):
            # 1. State karışımı (ortak uzayda)
            x_bar = np.zeros(7)
            for i, src_model in enumerate(self.models):
                x_i = project_to_common(src_model.x, src_model.model_type)
                x_bar += self.omega[i, j] * x_i

            # 2. Kovaryans karışımı
            P_bar = np.zeros((7, 7))
            for i, src_model in enumerate(self.models):
                x_i = project_to_common(src_model.x, src_model.model_type)
                P_i = project_cov_to_common(src_model.P, src_model.model_type)

                diff = (x_i - x_bar).reshape(-1, 1)
                P_bar += self.omega[i, j] * (P_i + diff @ diff.T)

            # 3. Hedef modele geri indir
            x0 = expand_from_common(x_bar, model.model_type)

            if model.model_type == "CV":
                P0 = P_bar[:4, :4]
            elif model.model_type == "CA":
                P0 = P_bar[:6, :6]
            elif model.model_type == "CT":
                P0 = np.zeros((5, 5))
                P0[:4, :4] = P_bar[:4, :4]
                P0[:4, 4] = P_bar[:4, 6]
                P0[4, :4] = P_bar[6, :4]
                P0[4, 4] = P_bar[6, 6]
            else:
                raise ValueError("Unknown model type")

            mixed_x.append(x0)
            mixed_P.append(P0)

        return mixed_x, mixed_P

    # -------------------
    # 3. Predict
    # -------------------
    def predict(self):
        self._compute_mixing_probabilities()
        mixed_x, mixed_P = self._mix_states()

        # filtrelere ata ve predict et
        for j, model in enumerate(self.models):
            model.x = mixed_x[j]
            model.P = mixed_P[j]
            model.predict()

        # IMM state estimate (öncül)
        self._compute_state_estimate()

    # -------------------
    # 4. Update
    # -------------------
    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        likelihoods = []
        for j, model in enumerate(self.models):
            # olabilirlik, update'ten ÖNCEKİ inovasyonla hesaplanmalı
            S = model.H @ model.P @ model.H.T + model.R
            y = z - model.H @ model.x
            try:
                like = np.exp(-0.5 * y.T @ np.linalg.solve(S, y)) / np.sqrt(np.linalg.det(2*np.pi*S))
            except np.linalg.LinAlgError:
                like = 1e-12
            likelihoods.append(float(like))

            model.update(z)

        self.likelihood = np.array(likelihoods)

        # model olasılıklarını güncelle
        self.mu = self.cbar * self.likelihood
        if np.sum(self.mu) <= 0 or np.any(np.isnan(self.mu)):
            self.mu = np.ones_like(self.mu) / self.N
        else:
            self.mu /= np.sum(self.mu)

        # IMM state estimate (güncel)
        self._compute_state_estimate()

    # -------------------
    # 5. State Estimate
    # -------------------
    def _compute_state_estimate(self):
        """IMM'nin karışık state tahminini güncelle"""
        self.x = np.zeros(7)
        for j, model in enumerate(self.models):
            self.x += self.mu[j] * project_to_common(model.x, model.model_type)

        self.P = np.zeros((7, 7))
        for j, model in enumerate(self.models):
            xj = project_to_common(model.x, model.model_type)
            Pj = project_cov_to_common(model.P, model.model_type)
            diff = (xj - self.x).reshape(-1, 1)
            self.P += self.mu[j] * (Pj + diff @ diff.T)

    # -------------------
    # 6. Kolay kullanım
    # -------------------
    def step(self, z):
        self.predict()
        self.update(z)
        return self.x[:4].copy(), self.mu  # [x, y, vx, vy] ve mode olasılıkları