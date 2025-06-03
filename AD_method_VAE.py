# pip install numpy pandas matplotlib scikit-learn torch PyQt5 tqdm
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

# VAE Encoder
class Encoder(torch.nn.Module):
    def __init__(self, input_size=140, latent_dim=8):
        super(Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc_mu = torch.nn.Linear(16, latent_dim)
        self.fc_logvar = torch.nn.Linear(16, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# VAE Decoder
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim=8, output_size=140):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, 16)
        self.fc2 = torch.nn.Linear(16, 32)
        self.fc3 = torch.nn.Linear(32, output_size)

    def forward(self, z):
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))
        return z

# VAE Model
class VAE(torch.nn.Module):
    def __init__(self, input_size=140, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, latent_dim)
        self.decoder = Decoder(latent_dim, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# VAE loss function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = torch.nn.functional.l1_loss(recon_x, x, reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

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

# Replace Autoencoder with VAE
vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters())
epochs = 100
batch_size = 32
losses = []

# Training loop
vae.train()
n_batches = len(normal_train_data) // batch_size
with trange(epochs) as tbar:
    for epoch in tbar:
        epoch_loss = 0.
        for i in range(0, len(normal_train_data), batch_size):
            batch = normal_train_data[i:i+batch_size]
            optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
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
plt.title("VAE Training Loss Curve")
plt.legend()
plt.savefig("vae_training_loss_curve.png")
plt.show()

# Evaluation
vae.eval()
with torch.no_grad():
    recon_test, mu, logvar = vae(test_data)
    reconstruction_error = torch.mean(torch.abs(recon_test - test_data), dim=1).cpu().numpy()
    recon_train, mu_train, logvar_train = vae(normal_train_data)
    train_error = torch.mean(torch.abs(recon_train - normal_train_data), dim=1).cpu().numpy()

threshold = np.percentile(train_error, 95)
y_pred = (reconstruction_error < threshold).astype(int)
y_true = test_labels.astype(int)

print(classification_report(y_true, y_pred, target_names=["Abnormal", "Normal"]))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Abnormal", "Normal"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("vae_confusion_matrix.png")
plt.show()


