import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths for data, model, logs and plots.
dataDir = './Alien-Plants-Analyzer/processed_data_NEW/invasive'
log_path = './Alien-Plants-Analyzer/modelTraining/final/trainLog.md'
model_path = './Alien-Plants-Analyzer/modelTraining/final/model.pth'
plot_path = './Alien-Plants-Analyzer/modelTraining/final/loss_plot.png'

# Ensure dataset exists
if not os.path.exists(dataDir):
    raise FileNotFoundError("invasive/ folder not found in processed_data_NEW")

# Transforms
trainTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

testTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
fullDataset = ImageFolder('./Alien-Plants-Analyzer/processed_data_NEW', transform=trainTransforms)

# Keep only invasive samples
invasive_idx = [i for i, label in enumerate(fullDataset.targets) if fullDataset.classes[label] == 'invasive']
fullDataset = Subset(fullDataset, invasive_idx)

print(f"Total invasive samples: {len(fullDataset)}")

# Train/val/test split
indices = list(range(len(fullDataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

trainSet = Subset(fullDataset, train_idx)
valSet = Subset(fullDataset, val_idx)
testSet = Subset(fullDataset, test_idx)

valSet.dataset = ImageFolder('./Alien-Plants-Analyzer/processed_data_NEW', transform=testTransforms)
testSet.dataset = ImageFolder('./Alien-Plants-Analyzer/processed_data_NEW', transform=testTransforms)

# DataLoaders
trainLoader = DataLoader(trainSet, batch_size=32, shuffle=True)
valLoader = DataLoader(valSet, batch_size=32, shuffle=False)
testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(device)

# Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
numEpochs = 30
history = {"train_loss": [], "val_loss": []}

for epoch in range(numEpochs):
    model.train()
    running_loss = 0.0
    for images, _ in trainLoader:  # labels ignored, always invasive
        images = images.to(device)
        labels = torch.ones(images.size(0), 1).to(device)  # all invasive
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in valLoader:
            images = images.to(device)
            labels = torch.ones(images.size(0), 1).to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    history["train_loss"].append(running_loss / len(trainLoader))
    history["val_loss"].append(val_loss / len(valLoader))

    print(f"Epoch {epoch+1}/{numEpochs} - Train Loss: {running_loss/len(trainLoader):.4f} - Val Loss: {val_loss/len(valLoader):.4f}")

# Save model
torch.save(model.state_dict(), model_path)

# Plot loss
plt.figure(figsize=(6, 4))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve (Invasive Only)")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path)

print(f"Training finished. Model saved to {model_path}, Loss plot saved to {plot_path}")

# Final Evaluation on Test Set (loss only)
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, _ in testLoader:
        images = images.to(device)
        labels = torch.ones(images.size(0), 1).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Final Test Loss: {test_loss/len(testLoader):.4f}")
