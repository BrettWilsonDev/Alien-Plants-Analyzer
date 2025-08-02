import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# WE CAN ALSO IMPLEMENT SINGLE INVASIVE AND NATIVE FOLDER WHERE IMAGES
# ARE AUTOMATICALLY SORTED INTO TEST VAL AND TRAIN DATASETS. 
# VALIDATION CAN BE GOOD FOR TESTING AND TESTING OVERFITTING ETC.

# Updated directories
trainDir = './processed_data/train'
testDir = './processed_data/test'
log_path = './modelTraining/final/trainLog.md'
model_path = './modelTraining/final/model.pth'
conf_matrix_path = './modelTraining/final/confusion_matrix.png'

# Check directories
if not os.path.exists(trainDir) or not os.path.exists(testDir):
    raise FileNotFoundError("Check your folder structure. Required: train/invasive, train/non-invasive, test/invasive, test/non-invasive.")

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

# Load datasets
trainData = ImageFolder(trainDir, transform=trainTransforms)
testData = ImageFolder(testDir, transform=testTransforms)

# Balance classes using WeightedRandomSampler
targets = [label for _, label in trainData]
class_counts = Counter(targets)
class_weights = [1.0 / class_counts[label] for label in targets]
sampler = WeightedRandomSampler(class_weights, len(class_weights))

# Dataloaders
trainLoader = DataLoader(trainData, batch_size=32, sampler=sampler)
testLoader = DataLoader(testData, batch_size=32, shuffle=False)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
numClasses = len(trainData.classes)
print(f'Classes: {trainData.classes}, Detected: {numClasses}')

# Model
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for param in model.features.parameters():
    param.requires_grad = False
for param in model.features[-3:].parameters():  # unfreeze last few layers
    param.requires_grad = True
model.classifier[1] = nn.Linear(model.classifier[1].in_features, numClasses)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
numEpochs = 10
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
    print(f"Epoch {epoch+1}/{numEpochs}, Loss: {running_loss/len(trainLoader):.4f}")

# Evaluation
model.eval()
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
for idx, class_name in enumerate(trainData.classes):
    print(f"Class: {class_name:15s}  Precision: {precisionPerClass[idx]:.2f}  Recall: {recallPerClass[idx]:.2f}")
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precisionMacro:.2f}, Macro Recall: {recallMacro:.2f}")

# Save metrics to markdown
os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, 'w') as f:
    f.write("# Training Summary\n\n")
    f.write(f"**Epochs:** {numEpochs}\n")
    f.write(f"**Final Accuracy:** {accuracy * 100:.2f}%\n")
    f.write(f"**Macro Precision:** {precisionMacro:.2f}\n")
    f.write(f"**Macro Recall:** {recallMacro:.2f}\n\n")
    f.write("## Per-Class Metrics\n")
    f.write("| Class | Precision | Recall |\n")
    f.write("|-------|-----------|--------|\n")
    for idx, class_name in enumerate(trainData.classes):
        f.write(f"| {class_name} | {precisionPerClass[idx]:.2f} | {recallPerClass[idx]:.2f} |\n")

print(f"\nMetrics saved to: {log_path}")

# Save model
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
df_cm = pd.DataFrame(conf_matrix.numpy(), index=trainData.classes, columns=trainData.classes)
sns.heatmap(df_cm, annot=True, fmt='.0f', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(conf_matrix_path)
print(f"Confusion matrix saved to: {conf_matrix_path}")
