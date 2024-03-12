# AI-ML-Assesment---Varahi

Metal Identification using Deep Learning
This repository contains code for a computer vision task aimed at identifying metal types: Copper, Brass, Steel, and Non-Metal. The project utilizes deep learning techniques for image classification, combining elements of data preprocessing, model training, and evaluation.

Task Overview
Task Description
Objective: Identification of Copper, Brass, Steel, and Non-Metal objects in images.
Techniques: Deep Learning/Computer Vision, Image Classification.
Tools: TensorFlow, Keras, VGG16 model.
Workflow
Data Collection: Collect images from a zip file containing samples of Copper, Brass, Steel, and Non-Metal objects.
Annotation: Annotate images by categorizing them into Copper, Brass, Steel, or Non-Metal classes.
Data Splitting: Split the annotated dataset into training, testing, and validation sets.
Model Training: Train a deep learning model using the VGG16 architecture to classify metal objects.
Model Evaluation: Evaluate the model's performance on the test and validation sets using relevant metrics.
GitHub Repository: Create a private GitHub repository with organized code and documentation.

Code Overview

Data Preparation
The create_test_dataset function creates a test dataset from a portion of the training dataset.

# Example Paths and Parameters
combined_train_dataset_path = 'path/to/training_set'
test_dataset_path = 'path/to/test_set'
img_width, img_height = 224, 224
batch_size = 32
test_percentage = 0.2

# Create test dataset from training dataset
create_test_dataset(combined_train_dataset_path, test_dataset_path, test_percentage)

Data Generators
Data generators are created for training and validation using the ImageDataGenerator from TensorFlow.

# Create data generators for training and validation with a split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Combine all training datasets into one training generator
train_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    combined_train_dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Create data generator for validation
validation_generator = datagen.flow_from_directory(
    combined_train_dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

Model Architecture and Training
A simple CNN model using the VGG16 architecture is built and trained.

# Build a simple CNN model using VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
base_model.trainable = False

model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))  # 4 classes: copper, brass, still, non-metal

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

Model Evaluation
The model is evaluated on the test set, and accuracy is printed.

# Evaluate the model on the test set
test_generator = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_directory(
    test_dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_eval = model.evaluate(test_generator)
print("Test Accuracy:", test_eval[1])

Instructions
Clone Repository:
Clone this repository to your local machine:

git clone <repository-url>
cd <repository-folder>

Data Preparation:
Follow the provided code to organize and prepare your dataset.
Training:
Train the model by running the code in a Python environment.
Evaluation:
Evaluate the model on the test set to assess its accuracy.
Documentation:
Explore the code comments for detailed explanations.
Refer to the README file for additional information.
