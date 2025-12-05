# -*- coding: utf-8 -*-
"""
Demo of logistic regression on mean and standard deviation of each sensor
for activity recognition data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import mode
import xgboost as xgb

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
# Last row of training data for train/test split
train_end_index = 3511



import numpy as np
import xgboost as xgb
from scipy.stats import mode

G = 9.80665
def predict_test(train_data, train_labels, test_data):
    # -------------------------------
    # Map labels to 0-based for XGBoost
    # -------------------------------
    unique_labels = np.unique(train_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    inv_label_map = {v: k for k, v in label_map.items()}
    train_labels_mapped = np.array([label_map[y] for y in train_labels])

    # -------------------------------
    # Gravity removal
    # -------------------------------
    def remove_accel(X, Y, Z, Wx, Wy, Wz, dt=1.0, beta=0.96):
        N, T = X.shape
        A_lin = np.zeros((N, T, 3))
        for n in range(N):
            a = np.column_stack([X[n], Y[n], Z[n]])
            w = np.column_stack([Wx[n], Wy[n], Wz[n]])
            g_b = np.zeros((T,3))
            g_b[0] = G * a[0] / (np.linalg.norm(a[0]) + 1e-9)
            for i in range(1,T):
                theta = np.linalg.norm(-w[i]*dt)
                k = -w[i]*dt/theta if theta > 1e-12 else -w[i]*dt
                v_par = np.dot(g_b[i-1], k) * k if theta > 1e-12 else 0
                v_perp = g_b[i-1] - v_par if theta > 1e-12 else g_b[i-1]
                g_pred = v_par + v_perp*np.cos(theta) + np.cross(k, v_perp)*np.sin(theta) if theta > 1e-12 else g_b[i-1]
                g_meas = G * a[i] / (np.linalg.norm(a[i])+1e-9)
                g_hat = beta * g_pred + (1 - beta) * g_meas
                g_b[i] = g_hat * (G / (np.linalg.norm(g_hat)+1e-9))
            A_lin[n] = a - g_b
        return A_lin[:,:,0], A_lin[:,:,1], A_lin[:,:,2]

    # -------------------------------
    # Extract features per window
    # -------------------------------
    def extract_features_window(window):
        ax, ay, az, gx, gy, gz = [window[:, i] for i in range(window.shape[1])]

        def stats(v):
            mean = np.mean(v)
            std = np.std(v)
            min_v = np.min(v)
            max_v = np.max(v)
            rng = max_v - min_v
            energy = np.mean(v ** 2)
            median = np.median(v)
            q25 = np.percentile(v, 25)
            q75 = np.percentile(v, 75)
            iqr = q75 - q25
            skew = 0 if std == 0 else ((np.mean((v - mean) ** 3)) / (std ** 3))
            kurt = 0 if std == 0 else ((np.mean((v - mean) ** 4)) / (std ** 4)) - 3
            rms = np.sqrt(np.mean(v ** 2))
            var = std ** 2
            return [mean, std, min_v, max_v, rng, energy, median, iqr, skew, kurt, rms, var]

        feats = []
        for v in (ax, ay, az, gx, gy, gz):
            feats.extend(stats(v))
        # Magnitudes
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
        feats.extend(stats(acc_mag))
        feats.extend(stats(gyr_mag))
        return np.array(feats, dtype=float)

    def build_feature_matrix_3d(data_3d):
        N = data_3d.shape[0]
        # Remove gravity per example
        X_lin, Y_lin, Z_lin = remove_accel(
            data_3d[:,:,0], data_3d[:,:,1], data_3d[:,:,2],
            data_3d[:,:,3], data_3d[:,:,4], data_3d[:,:,5]
        )
        Wx, Wy, Wz = data_3d[:,:,3], data_3d[:,:,4], data_3d[:,:,5]
        data_corrected = np.stack([X_lin,Y_lin,Z_lin,Wx,Wy,Wz], axis=2)
        return np.vstack([extract_features_window(data_corrected[i]) for i in range(N)])

    # -------------------------------
    # Build feature matrices
    # -------------------------------
    X_train = build_feature_matrix_3d(train_data)
    X_test = build_feature_matrix_3d(test_data)

    # -------------------------------
    # Majority vote smoothing
    # -------------------------------
    def smooth_predictions(y_pred, window_size=4):
        smoothed = np.copy(y_pred)
        half_window = window_size//2
        N = len(y_pred)
        for i in range(N):
            start = max(0, i-half_window)
            end = min(N, i+half_window+1)
            smoothed[i] = mode(y_pred[start:end], keepdims=False).mode
        return smoothed

    # -------------------------------
    # Class weights
    # -------------------------------
    classes, counts = np.unique(train_labels_mapped, return_counts=True)
    class_weights = {cls: len(train_labels_mapped)/c for cls,c in zip(classes,counts)}
    sample_weights = np.array([class_weights[y] for y in train_labels_mapped])

    # -------------------------------
    # XGBoost classifier
    # -------------------------------
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(unique_labels),
        n_estimators=600,
        max_depth=3,
        learning_rate=0.1,
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    clf.fit(X_train, train_labels_mapped, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    y_pred_smoothed = smooth_predictions(y_pred, window_size=4)
    # Map back to original labels
    y_pred_original = np.array([inv_label_map[y] for y in y_pred_smoothed])

    return y_pred_original


    


# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    # Load labels and training sensor data into 3-D array
    labels = np.loadtxt('labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt(sensor_names[0] + '_train_1.csv',
                              delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + '_train_1.csv', delimiter=',')

    # Split into training and test by row index. Do not use a random split as
    # rows are not independent!
    train_data = data[:train_end_index+1, :, :]
    train_labels = labels[:train_end_index+1]
    test_data = data[train_end_index+1:, :, :]
    test_labels = labels[train_end_index+1:]
    test_outputs = predict_test(train_data, train_labels, test_data)
    
    # Compute micro and macro-averaged F1 scores
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    print(f'Micro-averaged F1 score: {micro_f1}')
    print(f'Macro-averaged F1 score: {macro_f1}')
    
    # Examine outputs compared to labels
    n_test = test_labels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_test), test_labels, 'b.')
    plt.xlabel('Time window')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n_test), test_outputs, 'r.')
    plt.xlabel('Time window')
    plt.ylabel('Output (predicted target)')
    plt.show()
    