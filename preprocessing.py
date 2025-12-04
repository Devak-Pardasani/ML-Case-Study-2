import numpy as np

G = 9.81

def load_and_preprocess():
    """Load two datasets (train and test), remove gravity, extract features, and return X_train, X_test, y_train, y_test."""
    
    # ---------------------
    # Load train dataset
    # ---------------------
    X_data_train = np.loadtxt("Train_1/Acc_x_train_1.csv", delimiter=",")
    Y_data_train = np.loadtxt("Train_1/Acc_y_train_1.csv", delimiter=",")
    Z_data_train = np.loadtxt("Train_1/Acc_z_train_1.csv", delimiter=",")
    Wx_data_train = np.loadtxt("Train_1/Gyr_x_train_1.csv", delimiter=",")
    Wy_data_train = np.loadtxt("Train_1/Gyr_y_train_1.csv", delimiter=",")
    Wz_data_train = np.loadtxt("Train_1/Gyr_z_train_1.csv", delimiter=",")
    labels_train = np.loadtxt("Train_1/labels_train_1.csv", delimiter=",") - 1

    # ---------------------
    # Load test dataset
    # ---------------------
    X_data_test = np.loadtxt("Train_2/Acc_x_train_2.csv", delimiter=",")
    Y_data_test = np.loadtxt("Train_2/Acc_y_train_2.csv", delimiter=",")
    Z_data_test = np.loadtxt("Train_2/Acc_z_train_2.csv", delimiter=",")
    Wx_data_test = np.loadtxt("Train_2/Gyr_x_train_2.csv", delimiter=",")
    Wy_data_test = np.loadtxt("Train_2/Gyr_y_train_2.csv", delimiter=",")
    Wz_data_test = np.loadtxt("Train_2/Gyr_z_train_2.csv", delimiter=",")
    labels_test = np.loadtxt("Train_2/labels_train_2.csv", delimiter=",") - 1

    # ---------------------
    # Remove gravity
    # ---------------------
    X_data_train, Y_data_train, Z_data_train = remove_accel(
        X_data_train, Y_data_train, Z_data_train,
        Wx_data_train, Wy_data_train, Wz_data_train
    )
    X_data_test, Y_data_test, Z_data_test = remove_accel(
        X_data_test, Y_data_test, Z_data_test,
        Wx_data_test, Wy_data_test, Wz_data_test
    )

    # ---------------------
    # Feature extraction
    # ---------------------
    X_train = build_feature_matrix(
        X_data_train, Y_data_train, Z_data_train,
        Wx_data_train, Wy_data_train, Wz_data_train
    )
    X_test = build_feature_matrix(
        X_data_test, Y_data_test, Z_data_test,
        Wx_data_test, Wy_data_test, Wz_data_test
    )

    y_train = labels_train.squeeze().astype(int)
    y_test = labels_test.squeeze().astype(int)

    return X_train, X_test, y_train, y_test


# ---------------------------------
# Feature extraction functions
# ---------------------------------


def extract_features(ax, ay, az, gx, gy, gz):
    """Compute mean, std, min, max, range, and energy for each axis and magnitude."""
    def stats(v):
        return [
            np.mean(v),
            np.std(v),
            np.min(v),
            np.max(v),
            np.max(v) - np.min(v),
            np.mean(v**2),
        ]
    feats = []
    for v in (ax, ay, az, gx, gy, gz):
        feats.extend(stats(v))
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    feats.extend(stats(acc_mag))
    feats.extend(stats(gyr_mag))
    return np.array(feats, dtype=float)

def build_feature_matrix(X_data, Y_data, Z_data, Wx_data, Wy_data, Wz_data):
    """Build feature matrix for all windows in dataset."""
    N, T = X_data.shape
    assert T == 60, f"Expected 60 timesteps, got {T}"
    return np.vstack([
        extract_features(
            X_data[i], Y_data[i], Z_data[i],
            Wx_data[i], Wy_data[i], Wz_data[i]
        )
        for i in range(N)
    ])


# ---------------------------------
# Gravity removal (remove_accel)
# ---------------------------------
def remove_accel(X_data, Y_data, Z_data, Wx_data, Wy_data, Wz_data,
                 dt=1.0, method="A", beta=0.96, tau=1.0):
    """Remove gravity from accelerometer data using complementary filter or LPF."""

    def _to_2d(a):
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            if a.shape[0] != 60:
                raise ValueError("Each 1D input must have length 60.")
            return a[None, :], True
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
        theta = np.linalg.norm(w_dt)
        if theta < 1e-12:
            return v.copy()
        k = w_dt / theta
        v_par = np.dot(v, k) * k
        v_perp = v - v_par
        return v_par + v_perp*np.cos(theta) + np.cross(k, v_perp)*np.sin(theta)

    A_lin = np.zeros((N, T, 3))

    if method.upper() == "A":
        for n in range(N):
            a = np.column_stack([ax[n], ay[n], az[n]])
            w = np.column_stack([gx[n], gy[n], gz[n]])
            g_b = np.zeros((T, 3))
            g_b[0] = G * a[0] / _safe_norm(a[0])
            for i in range(1, T):
                g_pred = _rodrigues_rotate(g_b[i-1], -w[i] * dt)
                g_meas = G * a[i] / _safe_norm(a[i])
                g_hat = beta * g_pred + (1.0 - beta) * g_meas
                g_b[i] = g_hat * (G / _safe_norm(g_hat))
            A_lin[n] = a - g_b
    elif method.upper() == "B":
        alpha = tau / (tau + dt)
        for n in range(N):
            a = np.column_stack([ax[n], ay[n], az[n]])
            g_b = np.zeros((T, 3))
            g_b[0] = a[0]
            for i in range(1, T):
                g_b[i] = alpha * g_b[i-1] + (1 - alpha) * a[i]
                g_b[i] *= (G / _safe_norm(g_b[i]))
            A_lin[n] = a - g_b
    else:
        raise ValueError('method must be "A" or "B"')

    X_lin = A_lin[:, :, 0]
    Y_lin = A_lin[:, :, 1]
    Z_lin = A_lin[:, :, 2]

    if was1_x: X_lin = X_lin[0]
    if was1_y: Y_lin = Y_lin[0]
    if was1_z: Z_lin = Z_lin[0]

    return X_lin, Y_lin, Z_lin
