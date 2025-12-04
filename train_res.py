import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

def train_resnet_fc(X_train, X_test, y_train, y_test):
    # -----------------------------
    # Hyperparameters
    # -----------------------------
    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 120
    HIDDEN_SIZE = 64
    NUM_BLOCKS = 2
    DROPOUT_RATE = 0.3
    VAL_SPLIT = 0.2  # fraction of training data for validation

    # -----------------------------
    # Determine number of classes dynamically
    # -----------------------------
    NUM_CLASSES = int(np.max(np.concatenate([y_train, y_test])) + 1)
    print(f"Detected {NUM_CLASSES} classes.")

    # -----------------------------
    # Normalize features
    # -----------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # fit on train
    X_test = scaler.transform(X_test)        # transform test

    # -----------------------------
    # Convert to PyTorch tensors
    # -----------------------------
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # -----------------------------
    # Split training into train/val
    # -----------------------------
    val_size = int(len(X_train) * VAL_SPLIT)
    train_size = len(X_train) - val_size
    train_ds, val_ds = random_split(TensorDataset(X_train, y_train), [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # -----------------------------
    # Define Residual Block
    # -----------------------------
    class ResidualBlockFC(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, input_size)
            self.bn2 = nn.BatchNorm1d(input_size)

        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.fc1(x)))
            out = self.bn2(self.fc2(out))
            out += identity
            out = self.relu(out)
            return out

    # -----------------------------
    # Define ResNet-FC
    # -----------------------------
    class ResNetFC(nn.Module):
        def __init__(self, input_size, num_classes, hidden_size=HIDDEN_SIZE, num_blocks=NUM_BLOCKS, dropout=DROPOUT_RATE):
            super().__init__()
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.bn_input = nn.BatchNorm1d(hidden_size)
            self.relu = nn.ReLU()
            self.blocks = nn.ModuleList([ResidualBlockFC(hidden_size, hidden_size) for _ in range(num_blocks)])
            self.dropout = nn.Dropout(dropout)
            self.fc_out = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = self.relu(self.bn_input(self.input_layer(x)))
            for block in self.blocks:
                x = block(x)
            x = self.dropout(x)
            x = self.fc_out(x)
            return x

    # -----------------------------
    # Class weights
    # -----------------------------
    classes, counts = np.unique(y_train.numpy(), return_counts=True)
    class_weights = torch.tensor([len(y_train)/c for c in counts], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # -----------------------------
    # Device, model, optimizer
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetFC(input_size=X_train.shape[1], num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -----------------------------
    # Training loop with val monitoring
    # -----------------------------
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE = 10

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
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item() * xb.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # -----------------------------
    # Evaluation on test set
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

    return model
