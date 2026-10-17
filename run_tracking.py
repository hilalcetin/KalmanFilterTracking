import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from cv_kf import CVKalmanFilter
from ca_kf import CAKalmanFilter
from ct_kf import CTKalmanFilter
from imm import IMM

def run_sequence(seq_folder, out_video="tracking_output.mp4", fps=20.0):
    gt_path = os.path.join(seq_folder, "groundtruth.txt")
    bboxes = np.loadtxt(gt_path, delimiter=",")
    centers = np.column_stack((bboxes[:,0] + bboxes[:,2]/2,
                               bboxes[:,1] + bboxes[:,3]/2))

    frame_files = sorted(glob.glob(os.path.join(seq_folder, "*.jpg")))
    if len(frame_files) == 0:
        raise RuntimeError("Klasörde .jpg frame bulunamadı")

    # IMM setup
    dt, q, r = 1.0, 0.1, 5.0
    cv = CVKalmanFilter(dt, q, r)
    ca = CAKalmanFilter(dt, q, r)
    ct = CTKalmanFilter(dt, q_pos=q, q_vel=q, r=r)
    models = [cv, ca, ct]

    # Karşılaştırma için IMM'den bağımsız tek-model filtreler
    cv_alone = CVKalmanFilter(dt, q, r)
    ca_alone = CAKalmanFilter(dt, q, r)

    PI = np.array([[0.90, 0.05, 0.05],
                   [0.05, 0.90, 0.05],
                   [0.05, 0.05, 0.90]])
    mu0 = np.array([1/3, 1/3, 1/3])
    imm = IMM(models, mu0, PI)

    # İlk ölçümle başlat
    cx0, cy0 = centers[0]
    for m in models + [cv_alone, ca_alone]:
        m.x[:2] = np.array([cx0, cy0], dtype=float)

    # Video writer
    first_frame = cv2.imread(frame_files[0])
    H, W = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  
    out = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    # Her model için tahminleri kaydet
    traj_est_cv = []
    traj_est_ca = []
    traj_est_imm = []
    traj_meas = []

    mus = []

    for i, f in enumerate(frame_files):
        frame = cv2.imread(f)
        if frame is None:
            continue
        if i >= len(bboxes):
            break

        x, y, w, h = bboxes[i]
        cx, cy = centers[i]
        z = np.array([cx, cy], dtype=float)
        if np.isnan(z).any():
            continue

        # Tek tek modeller tahmini
        est_cv = cv_alone.step(z)
        est_ca = ca_alone.step(z)
        est_imm, mu = imm.step(z)

        traj_est_cv.append(est_cv.copy())
        traj_est_ca.append(est_ca.copy())
        traj_est_imm.append(est_imm.copy())
        traj_meas.append(z.copy())
        mus.append(mu.copy())

        # --- çizimler ---
        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        cv2.circle(frame, (int(round(cx)), int(round(cy))), 3, (255,0,0), -1)

        ex = int(np.clip(round(est_imm[0]), 0, W-1))
        ey = int(np.clip(round(est_imm[1]), 0, H-1))
        cv2.drawMarker(frame, (ex, ey), (0,0,255),
                       markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=18, thickness=2, line_type=cv2.LINE_AA)

        txt = f"CV:{mu[0]:.2f}  CA:{mu[1]:.2f}  CT:{mu[2]:.2f}"
        cv2.putText(frame, txt, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("IMM Tracking", frame)
        out.write(frame)
        if cv2.waitKey(1000//int(max(fps,1))) & 0xFF == 27:
            break

    out.release()
    cv2.destroyAllWindows()

    traj_est_cv  = np.array(traj_est_cv)
    traj_est_ca  = np.array(traj_est_ca)
    traj_est_imm = np.array(traj_est_imm)
    traj_meas    = np.array(traj_meas)
    mus = np.array(mus)

    # --- RMS rate error hesaplama ---
    def rms_rate_per_frame(traj_est, traj_gt):
        rms_list = []
        for est, gt in zip(traj_est, traj_gt):
            vel_est = np.array(est[2:4])
            vel_gt = np.array(gt[2:4])
            rms_list.append(np.linalg.norm(vel_est - vel_gt))
        return rms_list

    # Trajeler groundtruth ile hizalanmalı
    gt_traj = np.zeros((len(traj_meas), 4))  # [x, y, vx, vy]
    gt_traj[:,0:2] = traj_meas
    # hızları finite difference ile hesaplayalım
    gt_traj[1:,2] = np.diff(traj_meas[:,0])
    gt_traj[1:,3] = np.diff(traj_meas[:,1])

    # RMS rate error
    rms_cv  = rms_rate_per_frame(traj_est_cv, gt_traj)
    rms_ca  = rms_rate_per_frame(traj_est_ca, gt_traj)
    rms_imm = rms_rate_per_frame(traj_est_imm, gt_traj)

    # --- Grafik ---
    time = np.arange(len(traj_meas))
    plt.figure(figsize=(10,5))
    plt.plot(time, rms_cv, label="CV")
    plt.plot(time, rms_ca, label="CA")
    plt.plot(time, rms_imm, label="IMM (CV+CA+CT)")
    plt.xlabel("Frame")
    plt.ylabel("RMS Rate Error (px/frame)")
    plt.title("RMS Rate Error vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()

    # --- Ortalama model olasılığı grafiği ---
    time = np.arange(len(mus))

    plt.figure(figsize=(10,5))
    plt.plot(time, mus[:,0], label="CV probability", linewidth=2)
    plt.plot(time, mus[:,1], label="CA probability", linewidth=2)
    plt.plot(time, mus[:,2], label="CT probability", linewidth=2)

    plt.xlabel("Frame")
    plt.ylabel("Model Probability")
    plt.title("IMM Model Probabilities vs Time")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    seq_folder = os.path.join("val", "GOT-10k_Val_000001")
    run_sequence(seq_folder)