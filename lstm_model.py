import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def train_lstm(X_train, X_test, y_train, y_test):
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 30
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2

    # -----------------------------
    # Determine number of classes dynamically
    # -----------------------------
    NUM_CLASSES = int(np.max(np.concatenate([y_train, y_test])) + 1)
    print(f"Detected {NUM_CLASSES} classes.")

    # -----------------------------
    # Convert to PyTorch tensors
    # -----------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32)  # (N, 6, 60)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # LSTM expects input: (batch, seq_len, features)
    # Currently X_train is (N, 6, 60), so we need to permute to (N, 60, 6)
    X_train = X_train.permute(0, 2, 1)
    X_test = X_test.permute(0, 2, 1)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Define LSTM model
    # -----------------------------
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size=6, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES):
            super(LSTMClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
            out = out[:, -1, :]    # take the last timestep
            out = self.fc(out)
            return out

    # -----------------------------
    # Setup device, model, loss, optimizer
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(input_size=6, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(EPOCHS):
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
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

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
