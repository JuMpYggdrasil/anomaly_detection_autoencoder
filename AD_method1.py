# https://adamoudad.github.io/posts/ecg-anomaly-detection/
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training.")

# Load and prepare data
dataframe = pd.read_csv('ecg.csv', header=None)
raw_data = dataframe.values

labels = raw_data[:, -1]
data = raw_data[:, 0:-1]

normal_ecg = data[labels == 1][16]
abnormal_ecg = data[labels == 0][5]

# Calculate mean ECG for each class
normal_ecg_mean = data[labels == 1].mean(axis=0)
abnormal_ecg_mean = data[labels == 0].mean(axis=0)

# Plot and save example ECGs
plt.plot(normal_ecg, label="Normal")
plt.plot(abnormal_ecg, label="Abnormal")
plt.legend()
plt.savefig("ecg_example.png")
plt.show()

plt.plot(normal_ecg_mean, label="Normal mean")
plt.plot(abnormal_ecg_mean, label="Abnormal mean")
plt.legend()
plt.savefig("ecg_mean.png")
plt.show()

# Encoder definition
class Encoder(torch.nn.Module):
    def __init__(self, input_size=140):
        super(Encoder, self).__init__()
        self.enc1 = torch.nn.Linear(input_size, 32)
        self.enc2 = torch.nn.Linear(32, 16)
        self.enc3 = torch.nn.Linear(16, 8)

    def forward(self, x):
        x = torch.relu(self.enc1(x))
        x = torch.relu(self.enc2(x))
        x = torch.relu(self.enc3(x))
        return x

# Decoder definition
class Decoder(torch.nn.Module):
    def __init__(self, output_size=140):
        super(Decoder, self).__init__()
        self.dec1 = torch.nn.Linear(8, 16)
        self.dec2 = torch.nn.Linear(16, 32)
        self.dec3 = torch.nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.dec1(x))
        x = torch.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

# Autoencoder using Encoder and Decoder
class Autoencoder(torch.nn.Module):
    def __init__(self, input_size=140):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(input_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data split function
def train_test_split(data, labels, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(data))
    split = int(len(data) * test_size)
    test_idx, train_idx = indices[:split], indices[split:]
    return data[train_idx], data[test_idx], labels[train_idx], labels[test_idx]

# Split and normalize data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)
min_val, max_val = train_data.min(), train_data.max()
train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

# Convert to tensors and move to device
train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

# Training setup
batch_size = 32
epochs = 100
losses = []

autoencoder = Autoencoder().to(device)
optimizer = torch.optim.Adam(autoencoder.parameters())
criterion = torch.nn.L1Loss()

n_batches = len(normal_train_data) // batch_size

# Training loop
autoencoder.train()
with trange(epochs) as tbar:
    for epoch in tbar:
        epoch_loss = 0.
        for i in range(0, len(normal_train_data), batch_size):
            batch = normal_train_data[i:i+batch_size]
            optimizer.zero_grad()
            outputs = autoencoder(batch)
            loss = criterion(outputs, batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        losses.append(epoch_loss)
        tbar.set_postfix(loss=epoch_loss / float(n_batches))

# Plot and save learning curve
plt.figure()
plt.plot(losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Training Loss Curve")
plt.legend()
plt.savefig("training_loss_curve.png")
plt.show()

# Evaluation
autoencoder.eval()
with torch.no_grad():
    reconstructed = autoencoder(test_data)
    reconstruction_error = torch.mean(torch.abs(reconstructed - test_data), dim=1).cpu().numpy()
    train_recon = autoencoder(normal_train_data)
    train_error = torch.mean(torch.abs(train_recon - normal_train_data), dim=1).cpu().numpy()

threshold = np.percentile(train_error, 95)
y_pred = (reconstruction_error < threshold).astype(int)
y_true = test_labels.astype(int)

print(classification_report(y_true, y_pred, target_names=["Abnormal", "Normal"]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Abnormal", "Normal"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")  # Save the confusion matrix plot
plt.show()


