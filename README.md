# KalmanFilter Tracking

This project tracks an object's centre point across a video (image sequence) with Kalman filters.
Three motion models run in parallel and are combined with an **IMM (Interacting Multiple Model)** estimator:

| Model | File | State | Use |
|---|---|---|---|
| **CV** – Constant Velocity | `cv_kf.py` | `[x, y, vx, vy]` | straight, steady motion |
| **CA** – Constant Acceleration | `ca_kf.py` | `[x, y, vx, vy, ax, ay]` | speeding up / slowing down |
| **CT** – Coordinated Turn (EKF) | `ct_kf.py` | `[x, y, vx, vy, ω]` | turning at a constant rate |

All models measure only the box centre `(x, y)`.

## How the IMM works (`imm.py`)

Each frame:

1. **Mixing** – mixing probabilities are computed from the model probabilities `μ` and the transition matrix `PI`.
2. **Interaction** – model states and covariances are moved into a common 7-D space `[x, y, vx, vy, ax, ay, ω]`, mixed, and projected back to each model.
3. **Predict** – every filter predicts one step.
4. **Likelihood + update** – each model's likelihood is computed from its *pre-update* innovation, then the filter is updated with the measurement.
5. **Model probabilities** – `μ` is updated from the likelihoods and the combined IMM estimate is formed.

`imm.step(z)` returns `[x, y, vx, vy]` and the current model probabilities.

## Installation

Python **3.10 or higher**.

```bash
python3 -m venv .venv
source .venv/bin/activate     # Linux/macOS
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Data

Download the **GOT-10k Validation** set:

- [GOT-10k Validation (Kaggle)](https://www.kaggle.com/datasets/abhimanyukarshni/got10k?resource=download-directory&select=val)
- Put it in a `val/` folder in the project directory, e.g. `val/GOT-10k_Val_000001/`

Each sequence folder contains `*.jpg` frames and `groundtruth.txt` (`x, y, w, h` per line).

## Usage

```bash
python run_tracking.py
```

By default this runs `val/GOT-10k_Val_000001`. To use another sequence, change `seq_folder` at the bottom of `run_tracking.py`.

Parameters (in `run_sequence`): `dt = 1.0`, process noise `q = 0.1`, measurement noise `r = 5.0`, and a transition matrix with 0.90 on the diagonal.

## Outputs

- **Live window + `tracking_output.mp4`** – green box: ground truth, blue dot: measurement, red cross: IMM estimate, with CV/CA/CT probabilities shown at the top. Press `Esc` to stop.
- **RMS rate error plot** – velocity error over time (px/frame) for standalone CV, standalone CA and the IMM.
- **Model probability plot** – how the IMM weights CV, CA and CT over time.

## Notes and limitations

- Measurements are the ground-truth box centres, not detector output, so this evaluates the filter, not a full detector + tracker pipeline.
- The velocity error is measured against a finite-difference of those same measurements, not a true velocity.
- The standalone CV and CA filters used in the plot are separate objects from the ones inside the IMM.
- On turning segments the IMM still tends to favour CA over CT. Mixing pulls `ω` towards zero on every step and CA's process noise is broad; tuning `q`, `PI`, or CT's `q_omega` can improve this.

## Changelog

- **Fixed** – CT model: position now turns in the same direction as velocity, and the Jacobian's `ω` column was corrected.
- **Fixed** – IMM likelihood is computed from the innovation *before* the update (previously after, which locked the IMM onto CT).
- **Fixed** – CT receives `q_pos`, `q_vel` and `r` by name (previously `r` was passed as `q_vel`).
- **Fixed** – standalone CV/CA comparison filters no longer share objects with the IMM (previously every frame was predicted and updated twice).
