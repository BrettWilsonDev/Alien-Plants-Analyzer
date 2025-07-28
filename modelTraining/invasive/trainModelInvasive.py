import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Define data directories
trainDir = './processed_invasive/train'
testDir = './processed_invasive/test'

# Verify directories exist
if not os.path.exists(trainDir):
    raise FileNotFoundError(f"Train directory not found: {trainDir}")
if not os.path.exists(testDir):
    raise FileNotFoundError(f"Test directory not found: {testDir}")

# Data transformations
trainTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

testTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data loaders
trainData = ImageFolder(trainDir, transform=trainTransforms)
testData = ImageFolder(testDir, transform=testTransforms)

trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
testLoader = DataLoader(testData, batch_size=32, shuffle=False)

# Model
numClasses = len(trainData.classes)  # Automatically detect number of classes
print(f'Number of classes detected should be 3: {numClasses}')
model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
model.classifier[1] = nn.Linear(model.classifier[1].in_features, numClasses)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters())

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainLoader):.4f}')

# Evaluation
model.eval()
correct, total = 0, 0
tp = torch.zeros(numClasses)
fp = torch.zeros(numClasses)
fn = torch.zeros(numClasses)
with torch.no_grad():
    for images, labels in testLoader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    for cls in range(numClasses):
        tp[cls] += ((predicted == cls) & (labels == cls)).sum().item()
        fp[cls] += ((predicted == cls) & (labels != cls)).sum().item()
        fn[cls] += ((predicted != cls) & (labels == cls)).sum().item()

precisionPerClass = tp / (tp + fp + 1e-8)
recallPerClass = tp / (tp + fn + 1e-8)

precisionMacro = precisionPerClass.mean().item()
recallMacro = recallPerClass.mean().item()
accuracy = correct / total

# Print per-class stats
print("\nPer-Class Precision and Recall:")
for idx, class_name in enumerate(trainData.classes):
    p = precisionPerClass[idx].item()
    r = recallPerClass[idx].item()
    print(f"Class: {class_name:15s}  Precision: {p:.2f}  Recall: {r:.2f}")

# Print overall stats
print(f"\nOverall Accuracy: {accuracy:.2f}")
print(f"Macro Precision: {precisionMacro:.2f}, Macro Recall: {recallMacro:.2f}")

# Save metrics to trainLog.md
log_path = './modelTraining/invasive/trainLogInvasive.md'
with open(log_path, 'w') as f:
    f.write("# Training Summary\n\n")
    f.write(f"**Epochs:** {numEpochs}\n")
    f.write(f"**Final Accuracy:** {accuracy:.2f}\n")
    f.write(f"**Macro Precision:** {precisionMacro:.2f}\n")
    f.write(f"**Macro Recall:** {recallMacro:.2f}\n\n")

    f.write("## Per-Class Metrics\n")
    f.write("| Class | Precision | Recall |\n")
    f.write("|-------|-----------|--------|\n")
    for idx, class_name in enumerate(trainData.classes):
        p = precisionPerClass[idx].item()
        r = recallPerClass[idx].item()
        f.write(f"| {class_name} | {p:.2f} | {r:.2f} |\n")

print(f"\nMetrics saved to: {log_path}")

# Save model
os.makedirs('./modelTraining/invasive/', exist_ok=True)
torch.save(model.state_dict(), './modelTraining/invasive/modelInvasive.pth')

# ENTER WITH CAUTION. THIS FUCKS EVERYTHING UP.
# WHEN PROPER SOLUTION IS FOUND THIS NEEDS TO BE DELETED.
# # Convert PyTorch to ONNX and ONNX to TensorFlow SavedModel and TensorFlow to TensorFlow.js
# dummy_input = torch.randn(1, 3, 224, 224).to(device)
# onnx_path = './modelTraining/invasive/model.onnx'
# torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)
# print(f"ONNX model exported to: {onnx_path}")
