import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import mode
import numpy as np
from scipy.stats import mode

import numpy as np
from scipy.stats import mode

def smooth_predictions(y_pred, window_size=4):
    """
    Apply a majority (mode) filter to smooth predicted class sequences.
    Ensures stable predictions over short fluctuations.

    Parameters
    ----------
    y_pred : array-like, shape (N,)
        Predicted class labels.
    window_size : int
        Size of the sliding window. Should be odd.

    Returns
    -------
    y_smoothed : ndarray, shape (N,)
        Smoothed class predictions.
    """
    y_pred = np.asarray(y_pred)
    N = len(y_pred)
    half_window = window_size // 2
    y_smoothed = np.empty_like(y_pred)

    for i in range(N):
        start = max(0, i - half_window)
        end = min(N, i + half_window + 1)
        # Use try-except in case of unexpected mode behavior
        try:
            y_smoothed[i] = mode(y_pred[start:end], keepdims=True).mode[0]
        except IndexError:
            # fallback: take current prediction
            y_smoothed[i] = y_pred[i]

    return y_smoothed



def train_xgb(X_train, X_test, y_train, y_test, feature_names=None, n_jobs=-1):
    # -----------------------------
    # Class weights
    # -----------------------------
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = {cls: len(y_train)/c for cls, c in zip(classes, counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])
    print(f"Class weights: {class_weights}")

    # -----------------------------
    # Base XGB classifier
    # -----------------------------
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=int(np.max(y_train) + 1),
        n_jobs=-1,
        eval_metric='mlogloss'
    )

    # -----------------------------
    # Simplified grid search
    # -----------------------------
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
    }

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # -----------------------------
    # Fit with sample weights
    # -----------------------------
    grid.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Best parameters: {grid.best_params_}")

    best_clf = grid.best_estimator_

    # -----------------------------
    # Feature importance
    # -----------------------------
    if feature_names is not None:
        importances = best_clf.feature_importances_
        for f, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            print(f"{f}: {imp:.4f}")

    # -----------------------------
    # Predict & smooth
    # -----------------------------


    y_pred = best_clf.predict(X_test)
    y_pred_smoothed = smooth_predictions(y_pred,window_size=i)

    print("\nClassification report:\n", classification_report(y_test, y_pred_smoothed))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_smoothed))

    return best_clf
