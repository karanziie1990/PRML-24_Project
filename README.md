# K-Nearest Neighbors (KNN) Image Classifier for CIFAR-10 Dataset

## Overview
This repository contains a K-Nearest Neighbors (KNN) classifier implemented in Python using the scikit-learn library. The classifier is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Introduction
K-Nearest Neighbors (KNN) is a simple yet powerful algorithm used for classification and regression tasks. It belongs to the family of instance-based, non-parametric learning algorithms. In KNN, the classification of a new data point is determined by the majority class of its 'k' nearest neighbors in the feature space.

## Requirements
- Python 3.x
- scikit-learn
- numpy
- HoG features

## Files
- `FeatureExtraction(HoG,CNN)+KNN.ipynb`: Python script for extracting HoG, CNN features of the dataset
- `Preprocessing,Feature_Extraction,Similarity_Computation.ipynb`: Python script for testing the training the KNN classifier.

## Performance
The performance of the KNN classifier on the CIFAR-10 test data is as follows:

- Accuracy: 12%

## Contributors
- [Ansh Mehta](https://github.com/AnshMehta1)

## Acknowledgements
- The CIFAR-10 dataset was obtained from [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html).
- Pattern Recognition and Machine Learning Course by Prof. Anand Mishra [https://github.com/anandmishra22]
