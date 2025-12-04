import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def train_nn(X_train, X_test, y_train, y_test, raw=True):
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 120
    DROPOUT = 0.3

    # -----------------------------
    # Determine number of classes dynamically
    # -----------------------------
    NUM_CLASSES = int(np.max(np.concatenate([y_train, y_test])) + 1)
    print(f"Detected {NUM_CLASSES} classes.")

    # -----------------------------
    # Convert to PyTorch tensors
    # -----------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, T, C) or (N, 6, 60)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # If raw, reshape to (N, C, T) for Conv1d
    if raw:
        if X_train.ndim == 3:  # (N, T, C)
            X_train = X_train.permute(0, 2, 1)  # (N, C, T)
            X_test = X_test.permute(0, 2, 1)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Define CNN for time-series
    # -----------------------------
    class TimeSeriesCNN(nn.Module):
        def __init__(self, num_channels=X_train.shape[1], num_classes=NUM_CLASSES, dropout=DROPOUT):
            super(TimeSeriesCNN, self).__init__()
            self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(32)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(128)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x).squeeze(-1)  # (batch, 128)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    # -----------------------------
    # Setup device, model, loss, optimizer
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # L2 regularization

    # -----------------------------
    # Training loop with validation
    # -----------------------------
    best_val_loss = float('inf')
    patience = 10
    trigger_times = 0

    # Split off 10% of training for validation
    val_split = int(0.1 * len(train_loader.dataset))
    if val_split > 0:
        train_data, val_data = torch.utils.data.random_split(train_loader.dataset, [len(train_loader.dataset)-val_split, val_split])
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    else:
        val_loader = train_loader

    for epoch in range(EPOCHS):
        # Training
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # -----------------------------
    # Evaluation
    # -----------------------------
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    print("Classification Report:\n", classification_report(y_test.numpy(), all_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), all_preds))
