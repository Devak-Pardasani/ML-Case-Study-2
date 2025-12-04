import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def smooth_predictions(y_pred, window_size=5):
    """
    Smooth predicted labels using a sliding window mode.
    """
    y_pred = np.array(y_pred)
    smoothed = np.zeros_like(y_pred)
    half_w = window_size // 2

    for i in range(len(y_pred)):
        start = max(0, i - half_w)
        end = min(len(y_pred), i + half_w + 1)
        smoothed[i] = int(np.bincount(y_pred[start:end]).argmax())  # most frequent class in window

    return smoothed

def train_lgb(X_train, X_test, y_train, y_test):
    # -----------------------------
    # Compute class weights
    # -----------------------------
    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = {cls: len(y_train)/c for cls, c in zip(classes, counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])
    print(f"Class weights: {class_weights}")

    # -----------------------------
    # Base LightGBM classifier
    # -----------------------------
    clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=int(np.max(y_train) + 1),
        n_jobs=-1
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
    # Fit model with sample weights
    # -----------------------------
    grid.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Best parameters from Grid Search: {grid.best_params_}")

    # -----------------------------
    # Predict and smooth
    # -----------------------------
    y_pred = grid.predict(X_test)
    y_pred_smoothed = smooth_predictions(y_pred, window_size=7)  # apply smoothing

    # -----------------------------
    # Evaluate
    # -----------------------------
    print("\nClassification report:\n", classification_report(y_test, y_pred_smoothed))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_smoothed))

    return grid.best_estimator_
