import torch
import numpy as np

G = 9.81

def main():
    X_data = np.loadtxt("Train_1/Acc_x_train_1.csv", delimiter=",")
    Y_data = np.loadtxt("Train_1/Acc_y_train_1.csv", delimiter=",")
    Z_data = np.loadtxt("Train_1/Acc_z_train_1.csv", delimiter=",")
    Wx_data = np.loadtxt("Train_1/Gyr_x_train_1.csv", delimiter=",")
    Wy_data = np.loadtxt("Train_1/Gyr_y_train_1.csv", delimiter=",")
    Wz_data = np.loadtxt("Train_1/Gyr_z_train_1.csv", delimiter=",")
    labels = np.loadtxt("Train_1/labels_train_1.csv", delimiter=",")

    X_data, Y_data, Z_data = remove_accel(X_data=X_data, Y_data=Y_data, Z_data=Z_data, Wx_data=Wx_data, Wy_data=Wy_data, Wz_data=Wz_data)

    for i in range(5212):
        ang, max_accel, max_ang, area = 0, 0, 0, 0
        for j in range(60):
            wx_mag = Wx_data[i,j]**2
            wy_mag = Wy_data[i,j]**2
            wz_mag = Wz_data[i,j]**2
            x_mag = X_data[i,j]**2
            y_mag = Y_data[i,j]**2
            z_mag = Z_data[i,j]**2
            curr_accel = np.sqrt(x_mag+y_mag+z_mag)
            curr_ang = np.sqrt(wx_mag+wy_mag+wz_mag)
            ang += curr_ang
            area += curr_accel
            if curr_accel > max_accel:
                max_accel = curr_accel
            if curr_ang > max_ang:
                max_ang = curr_ang

        print(f"{float(max_accel)}, {float(area)}, {float(max_ang)}, {float(ang)}")


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