# pip install numpy pandas matplotlib scikit-learn torch tk PyQt5
# dataset from:
# https://www.cs.ucr.edu/~eamonn/time_series_data_2018/
# http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt # for display use: pip install PyQt5
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report




# ============================
# Load and preprocess the data
# ============================
print("Loading ECG data...")
df = pd.read_csv("ecg.csv", header=None)  # or remove 'header=None' if already has headers
labels = df.iloc[:, -1].values
data = df.iloc[:, :-1].values


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

X_train = data_scaled[labels == 0]
X_test = data_scaled
y_test = labels

# =======================
# Define Autoencoder model
# =======================
class ECGAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(140, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 140)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# ========================
# Train the autoencoder
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU for training.")
model = ECGAutoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

print("Training autoencoder...")
epochs = 50
for epoch in range(epochs):
    model.train()
    output = model(X_train_tensor)
    loss = criterion(output, X_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# ========================
# Evaluate reconstruction
# ========================
print("Evaluating reconstruction...")
model.eval()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
with torch.no_grad():
    reconstructed = model(X_test_tensor).cpu().numpy()

mse = np.mean((X_test - reconstructed) ** 2, axis=1)

# Plot histograms of reconstruction error
plt.figure(figsize=(8, 4))
plt.hist(mse[y_test == 0], bins=50, alpha=0.6, label="Normal")
plt.hist(mse[y_test == 1], bins=50, alpha=0.6, label="Anomaly")
plt.xlabel("Reconstruction error")
plt.ylabel("Count")
plt.legend()
plt.title("ECG Reconstruction Error")
plt.tight_layout()
plt.savefig("reconstruction_error.png")

# os.startfile("reconstruction_error.png")


# =======================
# Anomaly classification
# =======================
threshold = np.percentile(mse[y_test == 0], 95)
print(f"Chosen threshold: {threshold:.6f}")

y_pred = (mse > threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))
