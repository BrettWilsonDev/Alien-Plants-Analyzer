import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


train_dir = '../processed/train'
test_dir = '../processed/test'


# Data generators
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='binary')
test_data = test_gen.flow_from_directory(test_dir, target_size=(224, 224), class_mode='binary')

# Model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train
model.fit(train_data, epochs=10, validation_data=test_data)

# Evaluate
loss, acc, precision, recall = model.evaluate(test_data)
print(f'Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}')

# Save model
model.save('model.h5')
