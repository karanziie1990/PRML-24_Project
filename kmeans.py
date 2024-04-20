import numpy as np
from skimage.feature import hog
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.models import Model
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# Reduce the training dataset size
x_train = x_train[:5000]
y_train = y_train[:5000]


# Preprocess images
x_train = x_train.astype('float32') / 255

# Extract HOG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        hog_feat = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
        hog_features.append(hog_feat)
    return np.array(hog_features)

hog_features = extract_hog_features(x_train)

# Extract CNN features
def extract_cnn_features(images):
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv2').output)
    features = model.predict(images)
    return features.reshape(features.shape[0], -1)

cnn_features = extract_cnn_features(x_train)

# Combine features
combined_features = np.hstack((hog_features, cnn_features))
combined_features = combined_features.astype(np.float64)

# Apply K-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(combined_features)



def retrieve_similar_images(query_image_path, features, kmeans, x_train, train_labels, n=5):
    # Load the query image
    query_image = np.array(Image.open(query_image_path))
    
    # Extract HOG features of the query image
    hog_feat = hog(query_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=True)
    # Extract CNN features of the query image
    cnn_feat = extract_cnn_features(np.expand_dims(query_image, axis=0))
    # Combine features of the query image
    query_features = np.hstack((hog_feat, cnn_feat.flatten()))
    
    # Predict the cluster of the query image
    cluster_label = kmeans.predict(query_features.reshape(1, -1))

    # Find indices of all images belonging to the same cluster
    cluster_indices = np.where(kmeans.labels_ == cluster_label)[0]

    # Compute distances between the query image and all other images in the cluster
    distances = np.linalg.norm(features[cluster_indices] - query_features, axis=1)

    # Sort indices based on distances
    sorted_indices = cluster_indices[np.argsort(distances)]

    # Return the top n similar images
    similar_indices = sorted_indices[:n]
    
    # Retrieve the labels for the similar images
    similar_labels = [train_labels[idx] for idx in similar_indices]
    
    # Get the label of the query image
    query_label = train_labels[0]  # Index 0 since we're taking the first image
    
    # Calculate accuracy based on label matching
  # Calculate accuracy based on label matching
    query_in_similar = 1 if 0 in similar_indices else 0  # Check if the query image itself is in the similar images
    accuracy = (sum([1 for label in similar_labels if label == query_label]) + query_in_similar) / n
        
    return similar_indices, accuracy
# Example usage with an image path as input
query_image_path = "/content/airplane_cifar10test.png"  # Replace with the path to your query image

similar_indices, accuracy = retrieve_similar_images(query_image_path, combined_features, kmeans, x_train, y_train)

# Display similar images
plt.figure(figsize=(10, 5))
for i, idx in enumerate(similar_indices):
    plt.subplot(1, len(similar_indices), i + 1)
    plt.imshow(x_train[idx])
    plt.axis('off')
plt.show()

print("Accuracy:", accuracy)
