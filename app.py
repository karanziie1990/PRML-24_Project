from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load the saved model
cnn = load_model("cnn_model.h5")

# Use the trained CNN model to make predictions on the test data
y_pred = cnn.predict(X_test)

# Convert predicted probabilities to predicted classes using argmax
y_pred_classes = [np.argmax(element) for element in y_pred]
y_pred_classes = np.array(y_pred_classes)

# Define class labels for CIFAR-10 dataset
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Function to plot a sample image with its corresponding class label
def plot_sample(X, y, index):
    plt.figure(figsize=(10, 2))
    plt.axis('off')  # Hide the axis
    plt.imshow(X[index])  # Display the image
    plt.xlabel(classes[y[index][0]])  # Access the scalar value by indexing the flattened array
    plt.annotate(classes[y[index][0]], xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=12)  # Add class label
    plt.tight_layout()  # Adjust subplot parameters to remove excess padding
    image_path = f'static/image_{index}.png'  # Save the plotted image to the static directory
    plt.savefig(image_path, bbox_inches='tight')  # Save the plotted image without excess whitespace
    plt.close()  # Close the plot to free up memory

app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_image():
    # Get the index of the image from the request data
    data = request.json
    index = int(data.get('queryNumber', 0))  

    # Plot the query image
    plot_sample(X_test, y_test, index)

    # Return the response with query image, similar images filenames, and accuracy
    return jsonify({
        'queryImage': f'static/image_{index}.png'
    })

@app.route('/retrieve',methods=['POST'])
def retrieve_images() :
   # Get the index of the image from the request data
    data = request.json
    index = int(data.get('queryNumber', 0)) 

    # Get the actual label of the chosen image
    actual_label = y_test[index][0]

    i = 0
    similar_images = []

    # Loop to randomly select and display 10 images
    while i < 10:
        # Generate a random number between 0 and 9999
        random_number = random.randint(0, 9999)

        # Check if the predicted class of the randomly selected image matches the actual label
        if y_pred_classes[random_number] == actual_label:
            plot_sample(X_test, y_test, random_number)  # Plot the randomly selected image
            similar_images.append(f'static/image_{random_number}.png')
            i = i + 1  # Increment the counter variable


    return jsonify({
        'similarImages': similar_images,
    })
   
if __name__ == "__main__":
  app.run(host="0.0.0.0",port=5000)
