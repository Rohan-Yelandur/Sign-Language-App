import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

# Define paths
data_dir = 'Data'  # Path to the directory containing subdirectories for each class
model_save_path = 'Model/keras_modelv2.h5'

# Image dimensions
img_height, img_width = 300, 300
batch_size = 32
epochs = 10

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the images to [0, 1] range
    shear_range=0.2,  # Apply shear transformation
    zoom_range=0.2,  # Apply zoom transformation
    horizontal_flip=True,  # Randomly flip images horizontally
    validation_split=0.2  # 20% of the data for validation
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    data_dir,  # Directory where the data is located
    target_size=(img_height, img_width),  # Resize images to the specified size
    batch_size=batch_size,  # Number of images to be yielded from the generator per batch
    class_mode='categorical',  # Return 2D one-hot encoded labels
    subset='training'  # Set as training data
)

# Validation data generator
validation_generator = train_datagen.flow_from_directory(
    data_dir,  # Directory where the data is located
    target_size=(img_height, img_width),  # Resize images to the specified size
    batch_size=batch_size,  # Number of images to be yielded from the generator per batch
    class_mode='categorical',  # Return 2D one-hot encoded labels
    subset='validation'  # Set as validation data
)

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(26, activation='softmax')  # 26 classes for 26 ASL letters
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the model
model.save(model_save_path)

print(f"Model saved to {model_save_path}")