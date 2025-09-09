import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Paths for data, model, logs and plots.
dataDir = './Alien-Plants-Analyzer/processed_data_NEW'
log_path = './Alien-Plants-Analyzer/modelTraining/final/trainLog.md'
model_path = './Alien-Plants-Analyzer/modelTraining/final/model.pth'
conf_matrix_path = './Alien-Plants-Analyzer/modelTraining/final/confusion_matrix.png'
plot_path = './Alien-Plants-Analyzer/modelTraining/final/loss_accuracy_plot.png'

# Ensure that the dataset exists.
if not os.path.exists(dataDir):
    raise FileNotFoundError("processed_data_NEW not found. Expected structure: invasive/, non-invasive/")

# Transforms

# resize images, apply random flips/rotations/colors (data augmentation - 
# creating new data items with existing data items by introducing variablility), 
# convert to tensor, normalize.
trainTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# resize, convert, normalize (no augmentation).
testTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset once from folders and expects subfolders to be treated as classes.
# Detect and ensure the number of classes, invasive and native.
fullDataset = ImageFolder(dataDir, transform=trainTransforms)
numClasses = len(fullDataset.classes)
print(f"Classes: {fullDataset.classes}, Detected: {numClasses}")

# Split dataset into train, val, test
# Split into 80% train, 20% tes, split train into train and validation. 
# Stratify ensures that equal class balance existis within splits.
indices = list(range(len(fullDataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=[fullDataset.targets[i] for i in indices], random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=[fullDataset.targets[i] for i in train_idx], random_state=42)

# Create dataset subsets.
trainSet = Subset(fullDataset, train_idx)
valSet = Subset(fullDataset, val_idx)
testSet = Subset(fullDataset, test_idx)

# Apply testTransforms to val/test sets
valSet.dataset = ImageFolder(dataDir, transform=testTransforms)
testSet.dataset = ImageFolder(dataDir, transform=testTransforms)

# Weighted sampler for class balance in training
# Count the amount of samples per class, Assign weights proportional to counts.
# Ensure balanced batches with WeightedRadomSampler.
train_targets = [fullDataset.targets[i] for i in train_idx]
class_counts = Counter(train_targets)
class_weights = [1.0 / class_counts[label] for label in train_targets]
sampler = WeightedRandomSampler(class_weights, len(class_weights))

# Dataloaders to provide data in batches of 32
trainLoader = DataLoader(trainSet, batch_size=32, sampler=sampler)
valLoader = DataLoader(valSet, batch_size=32, shuffle=False)
testLoader = DataLoader(testSet, batch_size=32, shuffle=False)

# Use GPU if available, otherwise use CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
# Load the pretrained MobileNetV2 model.
# Freeze most layers and fine tune the last layers.
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():  # fine-tune last layers
    param.requires_grad = True
model.classifier[1] = nn.Linear(model.classifier[1].in_features, numClasses)
model = model.to(device) # Change final layer to output the number of classes
# and then move the model to GPU or CPU (Device). 

# Loss and optimizer
# Loss: Function indicates how wrong the model predictions are, compares predicted
# class with the true label, the bigger the mismatch the bigger the loss, thus the objective
# is to have the number as small as possible. 
# Optimizer: Adjust the model's weights to reduce the loss, make small changes when upadating for
# slower and more stable learning.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training with Early Stopping
# Train up to 30 epochs.
# Stop early if validation accuracy doesn’t improve for 3 epochs.
# Track loss/accuracy history.
numEpochs = 30
patience = 3  # stop if no improvement in 3 epochs
best_val_acc = 0.0
epochs_no_improve = 0
history = {"train_loss": [], "val_loss": [], "val_acc": []}

# Training: forward pass then compute loss then backward pass then upadate the weights.
for epoch in range(numEpochs):
    model.train()
    running_loss = 0.0
    for images, labels in trainLoader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation: test on validation set and calculate the accuracy and loss.
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in valLoader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = correct / total

    history["train_loss"].append(running_loss/len(trainLoader))
    history["val_loss"].append(val_loss/len(valLoader))
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{numEpochs} - Train Loss: {running_loss/len(trainLoader):.4f} - Val Loss: {val_loss/len(valLoader):.4f} - Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f"New best model saved with val_acc={val_acc:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# Plot Training/Validation Curves
plt.figure(figsize=(10, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history["val_acc"], label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(plot_path)
print(f"Training/validation curves saved to: {plot_path}")

# Load best model
# Switch to evaluation mode.
# Loop over test set and count correct predictions, build confusion matrix, track tp, fp, fn.
model.load_state_dict(torch.load(model_path))
model.eval()

# Final Evaluation on Test Set
correct, total = 0, 0
tp = torch.zeros(numClasses)
fp = torch.zeros(numClasses)
fn = torch.zeros(numClasses)
conf_matrix = torch.zeros(numClasses, numClasses)

with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for t, p in zip(labels.view(-1), predicted.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
            if t == p:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

# Metrics
precisionPerClass = tp / (tp + fp + 1e-8)
recallPerClass = tp / (tp + fn + 1e-8)
precisionMacro = precisionPerClass.mean().item()
recallMacro = recallPerClass.mean().item()
accuracy = correct / total

# Display results
print("\nPer-Class Precision and Recall:")
for idx, class_name in enumerate(fullDataset.classes):
    print(f"Class: {class_name:15s}  Precision: {precisionPerClass[idx]:.2f}  Recall: {recallPerClass[idx]:.2f}")
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precisionMacro:.2f}, Macro Recall: {recallMacro:.2f}")

# Save metrics
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'w') as f:
    f.write("# Training Summary\n\n")
    f.write(f"**Best Validation Accuracy:** {best_val_acc*100:.2f}%\n")
    f.write(f"**Final Test Accuracy:** {accuracy * 100:.2f}%\n")
    f.write(f"**Macro Precision:** {precisionMacro:.2f}\n")
    f.write(f"**Macro Recall:** {recallMacro:.2f}\n\n")
    f.write("## Per-Class Metrics\n")
    f.write("| Class | Precision | Recall |\n")
    f.write("|-------|-----------|--------|\n")
    for idx, class_name in enumerate(fullDataset.classes):
        f.write(f"| {class_name} | {precisionPerClass[idx]:.2f} | {recallPerClass[idx]:.2f} |\n")

print(f"\nMetrics saved to: {log_path}")

# Save confusion matrix
plt.figure(figsize=(8, 6))
df_cm = pd.DataFrame(conf_matrix.numpy(), index=fullDataset.classes, columns=fullDataset.classes)
sns.heatmap(df_cm, annot=True, fmt='.0f', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to: {conf_matrix_path}")
