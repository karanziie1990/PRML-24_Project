# Image Retrieval using SVM Classifier and PCA
This part demonstrates the implementation of an image retrieval system using Support Vector Machine (SVM) classifier with Polynomial kernel and Principal Component Analysis (PCA) for dimensionality reduction. The system retrieves relevant images based on a query image input from the CIFAR-10 dataset.

### Description:
1. SVM Classifier: Trained using the CIFAR-10 dataset to classify images into one of the ten classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The SVM classifier is implemented using the SVC class from scikit-learn.
2. Principal Component Analysis (PCA): Used for dimensionality reduction to reduce the number of features while retaining most of the variance in the data. PCA is applied to the combined training and validation datasets to transform the high-dimensional image data into a lower-dimensional space.
3. Image Retrieval: Given a query image from the test set, the system retrieves relevant images from the training set based on the predicted class label of the query image. Cosine similarity is used as the similarity measure between the query image and relevant images.

## Contributor
- SAURAV SONI (https://github.com/sauravsoni6377)

## Acknowledgments
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Pattern Recognition and Machine Learning](https://github.com/anandmishra22) by Anand Mishra, IITJ
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
