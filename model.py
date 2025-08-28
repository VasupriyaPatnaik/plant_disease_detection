#Setting Up Kaggle API
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

#Downloading Dataset
!kaggle datasets download -d rashikrahmanpritom/plant-disease-recognition-dataset

#Extracting the Dataset
import zipfile
zip_ref = zipfile.ZipFile('/content/plant-disease-recognition-dataset.zip', 'r')
zip_ref.extractall('/content/plant_leaf_disease_predictor')
zip_ref.close()

#Count of images in each subdirectory
import os

def total_files(folder_path):
    num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return num_files

train_files_healthy = "/content/plant_leaf_disease_predictor/Train/Train/Healthy"
train_files_powdery = "/content/plant_leaf_disease_predictor/Train/Train/Powdery"
train_files_rust = "/content/plant_leaf_disease_predictor/Train/Train/Rust"

test_files_healthy = "/content/plant_leaf_disease_predictor/Test/Test/Healthy"
test_files_powdery = "/content/plant_leaf_disease_predictor/Test/Test/Powdery"
test_files_rust = "/content/plant_leaf_disease_predictor/Test/Test/Rust"

valid_files_healthy = "/content/plant_leaf_disease_predictor/Validation/Validation/Healthy"
valid_files_powdery = "/content/plant_leaf_disease_predictor/Validation/Validation/Powdery"
valid_files_rust = "/content/plant_leaf_disease_predictor/Validation/Validation/Rust"

print("Number of healthy leaf images in training set", total_files(train_files_healthy))
print("Number of powder leaf images in training set", total_files(train_files_powdery))
print("Number of rusty leaf images in training set", total_files(train_files_rust))

print("========================================================")

print("Number of healthy leaf images in test set", total_files(test_files_healthy))
print("Number of powder leaf images in test set", total_files(test_files_powdery))
print("Number of rusty leaf images in test set", total_files(test_files_rust))

print("========================================================")

print("Number of healthy leaf images in validation set", total_files(valid_files_healthy))
print("Number of powder leaf images in validation set", total_files(valid_files_powdery))
print("Number of rusty leaf images in validation set", total_files(valid_files_rust))

total_train_images = total_files(train_files_healthy) + total_files(train_files_powdery) + total_files(train_files_rust)
total_test_images = total_files(test_files_healthy) + total_files(test_files_powdery) + total_files(train_files_rust)
total_validation_images = total_files(valid_files_healthy) + total_files(valid_files_powdery) + total_files(valid_files_rust)

print("========================================================")

print("Total number of training images:", total_train_images)
print("Total number of test images:", total_test_images)
print("Total number of validation images:", total_validation_images)

# Displaying Sample Images
import matplotlib.pyplot as plt
from PIL import Image

# Define image paths
image_path_healthy = '/content/plant_leaf_disease_predictor/Train/Train/Healthy/800edef467d27c15.jpg'
image_path_rust = '/content/plant_leaf_disease_predictor/Train/Train/Rust/80f09587dfc7988e.jpg'
image_path_powdery = '/content/plant_leaf_disease_predictor/Train/Train/Powdery/8299723bc94df5a8.jpg'

# Open the images
img_healthy = Image.open(image_path_healthy)
img_rust = Image.open(image_path_rust)
img_powdery = Image.open(image_path_powdery)

# Create a figure with 3 subplots (side by side)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Display the healthy leaf image
axs[0].imshow(img_healthy)
axs[0].axis('off')  # Hide axes
axs[0].set_title('Healthy Leaf')

# Display the rust leaf image
axs[1].imshow(img_rust)
axs[1].axis('off')  # Hide axes
axs[1].set_title('Rust Leaf')

# Display the powdery leaf image
axs[2].imshow(img_powdery)
axs[2].axis('off')  # Hide axes
axs[2].set_title('Powdery Leaf')

# Show the plot
plt.show()

#Install or upgrade TensorFlow only
!pip install tensorflow --upgrade

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Keras is now part of TensorFlow
keras_version = tf.keras.__version__
print("Keras version:", keras_version)

# Data Augmentation and Preparation
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Stronger augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,          # Random rotation
    width_shift_range=0.1,      # Horizontal shift
    height_shift_range=0.1,     # Vertical shift
    brightness_range=[0.7, 1.3],# Vary brightness
    channel_shift_range=10      # Random color variation
)

# Validation data should only be rescaled
test_datagen = ImageDataGenerator(rescale=1./255)

# Training generator
train_generator = train_datagen.flow_from_directory(
    '/content/plant_leaf_disease_predictor/Train/Train',
    target_size=(225, 225),
    batch_size=32,
    class_mode='categorical'
)

# Validation generator
validation_generator = test_datagen.flow_from_directory(
    '/content/plant_leaf_disease_predictor/Test/Test',
    target_size=(225, 225),
    batch_size=32,
    class_mode='categorical'
)

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Load one sample image from your training folder
img_path = "/content/plant_leaf_disease_predictor/Test/Test/Rust/831abdc76c05e23d.jpg"  # <-- change this to any image path
img = load_img(img_path, target_size=(225, 225))  # same size as your generator
x = img_to_array(img)  
x = np.expand_dims(x, axis=0)  # add batch dimension

# Create the same ImageDataGenerator you used for training
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generate and plot 10 augmented images
plt.figure(figsize=(12, 6))
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.subplot(2, 5, i+1)  # 2 rows, 5 columns
    plt.imshow(batch[0])
    plt.axis('off')
    i += 1
    if i == 5:  # show 5 examples
        break
plt.show()

#CNN Architecture
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

model = Sequential()

# Block 1
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(225,225,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Block 2
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Block 3
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten and Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))   # Helps reduce overfitting
model.add(Dense(3, activation='softmax'))  # 3 classes: Healthy, Powdery, Rust

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'precision', 'recall'])

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Optional: compute class weights (dataset is fairly balanced, but this can help)
train_labels = train_generator.classes
class_weights = compute_class_weight('balanced',
                                     classes=np.unique(train_labels),
                                     y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,                        # increased epochs for better learning
    validation_data=validation_generator,
    class_weight=class_weights_dict,  # optional, helps Rust class
    callbacks=[early_stop, reduce_lr]
)

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

# Set up seaborn for styling
sns.set_theme()
sns.set_context("poster")

# Create a figure to plot all metrics
figure(figsize=(14, 10), dpi=100)

# Plot accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Accuracy',fontsize=18)
plt.legend(loc='upper left',fontsize=15)

# Plot precision
if 'precision' in history.history:
    plt.subplot(2, 2, 2)
    plt.plot(history.history['precision'], label='train_precision')
    plt.plot(history.history['val_precision'], label='val_precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch',fontsize=18)
    plt.ylabel('Precision',fontsize=18)
    plt.legend(loc='upper left',fontsize=15)

# Plot recall
if 'recall' in history.history:
    plt.subplot(2, 2, 3)
    plt.plot(history.history['recall'], label='train_recall')
    plt.plot(history.history['val_recall'], label='val_recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch',fontsize=18)
    plt.ylabel('Recall',fontsize=18)
    plt.legend(loc='upper left',fontsize=15)

# Plot loss
plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.legend(loc='upper left',fontsize=15)

# Adjust layout and show plots
plt.tight_layout()
plt.show()

# Saving the Model
model.save("model.h5")

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to preprocess the image
def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

# Load and preprocess the image
image_path = '/content/plant_leaf_disease_predictor/Validation/Validation/Powdery/9b6a318cc5721d73.jpg'
x = preprocess_image(image_path)

# Make predictions
predictions = model.predict(x)
predicted_class = np.argmax(predictions[0])

# 'train_generator' is defined and has the class indices
labels = train_generator.class_indices
labels = {v: k for k, v in labels.items()} # Invert the dictionary to map indices to labels

predicted_label = labels[predicted_class]

# Output the predicted label
print(f'Predicted Label: {predicted_label}')

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate predictions and true labels for the entire test dataset
def get_all_predictions_and_labels(generator):
    num_samples = generator.samples
    num_batches = num_samples // generator.batch_size
    all_predictions = []
    all_true_labels = []

    for _ in range(num_batches):
        images, labels = generator.__next__()
        predictions = model.predict(images)
        all_predictions.extend(np.argmax(predictions, axis=-1))
        all_true_labels.extend(np.argmax(labels, axis=-1))

    return np.array(all_true_labels), np.array(all_predictions)

# Get all predictions and true labels
true_labels, predicted_labels = get_all_predictions_and_labels(validation_generator)

# Compute the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Create a figure to plot the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_generator.class_indices.keys(),
            yticklabels=train_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Test Data')
plt.show()
