import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost

G = 9.81

def main():
    X_data = np.loadtxt("Train_1/Acc_x_train_1.csv", delimiter=",")
    Y_data = np.loadtxt("Train_1/Acc_y_train_1.csv", delimiter=",")
    Z_data = np.loadtxt("Train_1/Acc_z_train_1.csv", delimiter=",")
    Wx_data = np.loadtxt("Train_1/Gyr_x_train_1.csv", delimiter=",")
    Wy_data = np.loadtxt("Train_1/Gyr_y_train_1.csv", delimiter=",")
    Wz_data = np.loadtxt("Train_1/Gyr_z_train_1.csv", delimiter=",")
    labels = np.loadtxt("Train_1/labels_train_1.csv", delimiter=",")
    for i in range(len(labels)):
        labels[i] = labels[i] - 1
        

    X_data, Y_data, Z_data = remove_accel(X_data=X_data, Y_data=Y_data, Z_data=Z_data, Wx_data=Wx_data, Wy_data=Wy_data, Wz_data=Wz_data)
    def extract_features(ax, ay, az, gx, gy, gz):
        def stats(v):
            # mean, std, min, max, range, energy
            return [
                np.mean(v),
                np.std(v),
                np.min(v),
                np.max(v),
                np.max(v) - np.min(v),
                np.mean(v ** 2),
            ]

        feats = []

        # Per-channel features
        for v in (ax, ay, az, gx, gy, gz):
            feats.extend(stats(v))

        # Magnitudes
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        feats.extend(stats(acc_mag))
        feats.extend(stats(gyr_mag))

        return np.array(feats, dtype=float)

    # 2) Build feature matrix from all windows
    N, T = X_data.shape
    assert T == 60, f"Expected 60 timesteps, got {T}"

    X_feat = np.vstack([
        extract_features(
            X_data[i], Y_data[i], Z_data[i],
            Wx_data[i], Wy_data[i], Wz_data[i]
        )
        for i in range(N)
    ])

    y = labels.squeeze().astype(int)  # (N,)

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y,
        test_size=0.5,
        random_state=3,
        stratify=y
    )

    # 4) Random Forest classifier
    clf = xgboost.XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=1,
        gamma=0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))


def remove_accel(X_data, Y_data, Z_data, Wx_data, Wy_data, Wz_data,
                 dt=1.0, method="A", beta=0.96, tau=1.0):
    """
    Gravity removal. Returns linear acceleration in body frame with same shape as inputs.

    Inputs (each): shape (60,) or (N, 60)
      - X_data, Y_data, Z_data: accelerometer in m/s^2
      - Wx_data, Wy_data, Wz_data: gyro in rad/s  (convert deg/s to rad/s before calling if needed)

    Params:
      method: "A" (complementary, uses gyro) or "B" (LPF, accel-only)
      dt: sampling period in seconds (1.0 for 1 Hz)
      beta: complementary filter gyro trust (0.95..0.99)  [only for method A]
      tau: LPF time constant in seconds (e.g., 0.5..2.0)  [only for method B]

    Returns:
      X_lin, Y_lin, Z_lin  (same shape as inputs)
    """

    def _to_2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            if a.shape[0] != 60:
                raise ValueError("Each 1D input must have length 60.")
            return a[None, :], True   # (1,60), flag: was_1d
        if a.ndim == 2:
            if a.shape[1] != 60:
                raise ValueError("Each 2D input must have shape (N,60).")
            return a, False
        raise ValueError("Each input must be shape (60,) or (N,60).")

    ax, was1_x = _to_2d(X_data)
    ay, was1_y = _to_2d(Y_data)
    az, was1_z = _to_2d(Z_data)
    gx, _ = _to_2d(Wx_data)
    gy, _ = _to_2d(Wy_data)
    gz, _ = _to_2d(Wz_data)

    N, T = ax.shape

    def _safe_norm(v, eps=1e-9):
        n = np.linalg.norm(v)
        return max(n, eps)

    def _rodrigues_rotate(v, w_dt):
        """Rotate vector v by axis-angle w_dt (rad)."""
        theta = np.linalg.norm(w_dt)
        if theta < 1e-12:
            return v.copy()
        k = w_dt / theta
        v_par = np.dot(v, k) * k
        v_perp = v - v_par
        return v_par + v_perp*np.cos(theta) + np.cross(k, v_perp)*np.sin(theta)

    A_lin = np.zeros((N, T, 3))

    if method.upper() == "A":
        # Complementary (gyro + accel-direction)
        for n in range(N):
            a = np.column_stack([ax[n], ay[n], az[n]])
            w = np.column_stack([gx[n], gy[n], gz[n]])
            g_b = np.zeros((T, 3))
            g_b[0] = G * a[0] / _safe_norm(a[0])
            for i in range(1, T):
                g_pred = _rodrigues_rotate(g_b[i-1], -w[i] * dt)  # frame rotation
                g_meas = G * a[i] / _safe_norm(a[i])
                g_hat  = beta * g_pred + (1.0 - beta) * g_meas
                g_b[i] = g_hat * (G / _safe_norm(g_hat))
            A_lin[n] = a - g_b
    elif method.upper() == "B":
        # Pure LPF on accel (no gyro)
        alpha = tau / (tau + dt)
        for n in range(N):
            a = np.column_stack([ax[n], ay[n], az[n]])
            g_b = np.zeros((T, 3))
            g_b[0] = a[0]
            for i in range(1, T):
                g_b[i] = alpha * g_b[i-1] + (1 - alpha) * a[i]
                g_b[i] *= (G / _safe_norm(g_b[i]))  # keep magnitude near g
            A_lin[n] = a - g_b
    else:
        raise ValueError('method must be "A" or "B"')

    # Split back to axes
    X_lin = A_lin[:, :, 0]
    Y_lin = A_lin[:, :, 1]
    Z_lin = A_lin[:, :, 2]

    # If original inputs were 1D, squeeze back to 1D
    if was1_x: X_lin = X_lin[0]
    if was1_y: Y_lin = Y_lin[0]
    if was1_z: Z_lin = Z_lin[0]

    return X_lin, Y_lin, Z_lin

if __name__ == "__main__":
    main()