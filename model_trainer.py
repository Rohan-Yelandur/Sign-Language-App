from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import os

class ModelTrainer:
    def __init__(self, data_dir, model_save_path, img_size, batch_size, epochs):
        self.data_dir = data_dir
        self.model_save_path = model_save_path
        self.img_height = img_size
        self.img_width = img_size
        self.batch_size = batch_size
        self.epochs = epochs

    def train_model(self):
        # Data augmentation and normalization
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',  # Return 2D one-hot encoded labels
            subset='training',
            shuffle=True
        )

        # Validation data generator
        validation_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',  # Return 2D one-hot encoded labels
            subset='validation',
            shuffle=True
        )

        if train_generator.samples == 0 or validation_generator.samples == 0:
            raise ValueError("No images found in the specified directory. Please check the directory path and contents.")

        # Load the MobileNetV2 model, excluding the top layers
        base_model = MobileNetV2(input_shape=(self.img_height, self.img_width, 3), include_top=False, weights='imagenet')
        base_model.trainable = False

        # Define the model
        num_classes = train_generator.num_classes  # Number of classes
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')  # num_classes for multi-class classification
        ])

        # Compile the model
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Early stopping callback
        # Training will stop if the validation loss does not improve for 5 consecutive epochs.
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping]
        )

        # Save the model
        model.save(self.model_save_path)

        print(f"Model saved to {self.model_save_path}")