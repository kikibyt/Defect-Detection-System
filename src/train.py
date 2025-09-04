import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from model import Autoencoder
import os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from pytorch_msssim import SSIM  # pip install pytorch-msssim
import matplotlib.pyplot as plt

# -------------------------
# Custom Dataset
# -------------------------
class MVTecDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []  # 0: normal, 1: defect
        if train:
            for img_path in os.listdir(os.path.join(root_dir, 'train/good')):
                self.images.append(os.path.join(root_dir, 'train/good', img_path))
                self.labels.append(0)
        else:
            for subdir in ['test/good', 'test/broken_large']:
                label = 0 if 'good' in subdir else 1
                for img_path in os.listdir(os.path.join(root_dir, subdir)):
                    self.images.append(os.path.join(root_dir, subdir, img_path))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# -------------------------
# Data
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = MVTecDataset('bottle/bottle', train=True, transform=transform)
test_dataset = MVTecDataset('bottle/bottle', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------
# Setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
mse_loss = nn.MSELoss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, auc_scores = [], [], []
best_auc, best_state = 0, None

# Create directory for proofs
os.makedirs("proofs", exist_ok=True)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(50):
    # --- Training ---
    model.train()
    train_loss = 0
    for data, _ in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = mse_loss(output, data) + (1 - ssim_loss(output, data))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # --- Validation Loss (good only) ---
    model.eval()
    val_loss, count = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            mask = (label == 0)  # keep only good
            if mask.sum() == 0:
                continue
            data = data[mask].to(device)
            output = model(data)
            loss = mse_loss(output, data) + (1 - ssim_loss(output, data))
            val_loss += loss.item()
            count += 1
    val_loss /= max(1, count)
    val_losses.append(val_loss)

    # --- AUROC (all test images) ---
    errors, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            output = model(data)
            error = torch.mean((data - output) ** 2, dim=[1, 2, 3])
            errors.extend(error.cpu().numpy())
            labels.extend(label.numpy())
    auc = roc_auc_score(labels, errors)
    auc_scores.append(auc)

    if auc > best_auc:
        best_auc = auc
        best_state = model.state_dict()

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUROC: {auc:.4f}")

# -------------------------
# Save best model
# -------------------------
torch.save(best_state, "autoencoder_best.pth")
print(f"Best AUROC: {best_auc:.4f}")

# -------------------------
# Compute best threshold
# -------------------------
fpr, tpr, thresholds = roc_curve(labels, errors)
j_scores = tpr - fpr
best_threshold = thresholds[np.argmax(j_scores)]

print(f"Optimal threshold based on validation: {best_threshold:.6f}")
np.save("threshold.npy", best_threshold)

# -------------------------
# Plot and save curves
# -------------------------
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", color="#1f77b4")
plt.plot(val_losses, label="Val Loss (good only)", color="#ff7f0e")
plt.legend()
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("proofs/loss_curves.png")  # Save loss plot

plt.subplot(1, 2, 2)
plt.plot(auc_scores, label="AUROC", color="#2ca02c")
plt.legend()
plt.title("AUROC per Epoch")
plt.xlabel("Epoch")
plt.ylabel("AUROC")
plt.grid(True)
plt.savefig("proofs/auroc_curve.png")  # Save AUROC plot
plt.close()

# -------------------------
# Plot and save ROC curve
# -------------------------
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {best_auc:.4f})", color="#1f77b4")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("proofs/roc_curve.png")
plt.close()

# -------------------------
# Generate sample reconstructions
# -------------------------
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        output = model(data)
        for i in range(min(3, data.size(0))):  # Save first 3 images
            orig_img = data[i].cpu().numpy().squeeze()
            recon_img = output[i].cpu().numpy().squeeze()
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(orig_img, cmap='gray')
            plt.title("Original")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(recon_img, cmap='gray')
            plt.title("Reconstructed")
            plt.axis("off")
            plt.savefig(f"proofs/reconstruction_{i}.png")
            plt.close()
        break  # Process one batch

print("Image proofs saved in 'proofs/' directory")