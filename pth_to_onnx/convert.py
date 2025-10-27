import torch
import torchvision.models as models
import torch.nn as nn

num_classes = 2
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load("../models/model.pth", map_location="cpu")
state_dict = checkpoint.get("model_state_dict", checkpoint)

# Fix key mismatch if needed
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("fc.1"):
        new_key = k.replace("fc.1", "fc")
    else:
        new_key = k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
model.eval()

# Create a dummy input matches transform: 224x224 image
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model, 
    dummy_input, 
    "../models/model.onnx", 
    export_params=True, 
    opset_version=11,  # Compatible with ONNX Runtime Web
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output']
)

print("Model exported to model.onnx")