Model Training Summary
1. Model Information
Architecture Used: MobileNetV2 (weights='IMAGENET1K_V1')

Modification: Replaced final classification layer to output 2 classes (invasive, non-invasive)

Frozen Layers: All except last 3 layers of the feature extractor

Device: CUDA (GPU) if available, otherwise CPU

2. Dataset
Training Directory: ./processed_data/train

Testing Directory: ./processed_data/test

Classes: ['invasive', 'non-invasive']

Total Classes: 2

Data Balance: Handled with WeightedRandomSampler to address class imbalance in training data

3. Image Preprocessing Techniques
Training Transforms:
Resize to (224, 224)

Random horizontal flip

Random rotation (10 degrees)

Color jitter

Tensor conversion and normalization using ImageNet stats

Testing Transforms:
Resize to (224, 224)

Tensor conversion and normalization using ImageNet stats

4. Training Process
Epochs Trained: 10

Optimizer: Adam (lr=1e-4)

Loss Function: CrossEntropyLoss

Batch Size: 32

Sampler: WeightedRandomSampler (for balanced class representation)

Loss Trend:
yaml
Copy
Edit
Epoch 1:  0.3636
Epoch 2:  0.1360
Epoch 3:  0.1013
Epoch 4:  0.0573
Epoch 5:  0.0475
Epoch 6:  0.0495
Epoch 7:  0.0336
Epoch 8:  0.0446
Epoch 9:  0.0382
Epoch10:  0.0275
Accuracy and Metrics:
Overall Accuracy: 97.28%

Macro Precision: 0.97

Macro Recall: 0.96

Per-Class Metrics:
Class	Precision	Recall
Invasive	0.97	0.99
Non-Invasive	0.98	0.92

5. Final Model Details
Saved Model Path: ./modelTraining/final/model.pth

Confusion Matrix Path: ./modelTraining/final/confusion_matrix.png

Inference Speed: Fast, suitable for real-time use (based on MobileNetV2 efficiency)

Model Size: Lightweight; ideal for web deployment

6. Observations and Lessons
What Worked:
Using MobileNetV2 helped reduce training time while maintaining high accuracy.

Data augmentation improved model robustness.

Weighted sampling significantly helped with class imbalance.

Freezing most layers and unfreezing only the final few allowed efficient fine-tuning.

What Didn’t:
Without a validation split, it's hard to measure overfitting precisely. Future iterations should use a proper train/val/test split.

More aggressive regularization or early stopping may help generalization on unseen data.

