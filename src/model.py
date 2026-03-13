import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────
# LSTM Model
# ─────────────────────────────────────────
class EnergyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=24, dropout=0.2):
        super(EnergyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])   # Take last timestep
        return out


# ─────────────────────────────────────────
# Transformer Model (advanced option)
# ─────────────────────────────────────────
class EnergyTransformer(nn.Module):
    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, output_size=24, dropout=0.1):
        super(EnergyTransformer, self).__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                      dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])
        return x


# ─────────────────────────────────────────
# Dataset helper
# ─────────────────────────────────────────
def create_sequences(data: np.ndarray, lookback: int = 168, horizon: int = 24):
    """
    Create (X, y) sequences for time series.
    lookback = 168 hours (1 week of history)
    horizon  = 24 hours to predict ahead
    """
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────
def train_model(model, X_train, y_train, epochs=30, lr=1e-3, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}")

    print("✅ Training complete!")
    return model
