# Importing necessary libraries
from tensorflow.keras import datasets, layers, models  # Import necessary components from Keras
import numpy as np

# Separating the training and testing dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Define the architecture of the Convolutional Neural Network (CNN) using Sequential model
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),  # Convolutional layer with 32 filters and 3x3 kernel size, ReLU activation function, and input shape of (32, 32, 3)
    layers.MaxPooling2D((2, 2)),  # MaxPooling layer with pool size (2, 2)

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),  # Convolutional layer with 64 filters and 3x3 kernel size, ReLU activation function
    layers.MaxPooling2D((2, 2)),  # MaxPooling layer with pool size (2, 2)

    layers.Flatten(),  # Flatten layer to convert the 2D feature maps into a 1D vector
    layers.Dense(64, activation='relu'),  # Dense layer with 64 neurons and ReLU activation function
    layers.Dense(10, activation='softmax')  # Dense output layer with 10 neurons (for 10 classes) and softmax activation function
])

# Compile the CNN model with Adam optimizer, sparse categorical crossentropy loss function, and accuracy metric
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the CNN model using training data (X_train, y_train) for 10 epochs
cnn.fit(X_train, y_train, epochs=10)

# Save the model in HDF5 format
cnn.save("cnn_model.h5")
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)