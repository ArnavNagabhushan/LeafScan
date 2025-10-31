# -----------------------------------------------------------
# Train Plant Disease Detection Model - All Plants (Auto Split)
# Team: B1052JR2 | Developer: Arnav Nagabhushan
# -----------------------------------------------------------

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# -----------------------------------------------------------
# Paths to your dataset
# -----------------------------------------------------------
train_dir = "dataset/train"  # No valid folder needed

# -----------------------------------------------------------
# Data augmentation and automatic validation split
# -----------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 👈 Automatically split 80/20
)

# -----------------------------------------------------------
# Load training and validation data from same folder
# -----------------------------------------------------------
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # 80% data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # 20% data
)

# Get class labels dynamically
labels = list(train_generator.class_indices.keys())
print(f"✅ Classes found: {labels}")

# -----------------------------------------------------------
# Build the model using VGG16 as base
# -----------------------------------------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # ✅ fixes shape mismatch
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')  # dynamically set number of outputs
])

# Freeze VGG16 layers to speed up training
for layer in base_model.layers:
    layer.trainable = False

# -----------------------------------------------------------
# Compile the model
# -----------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------------
# Train the model
# -----------------------------------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=5
)

# -----------------------------------------------------------
# Save the trained model
# -----------------------------------------------------------
os.makedirs("model", exist_ok=True)
model.save("model/plant_model_all.h5")
print("✅ Model trained and saved successfully!")