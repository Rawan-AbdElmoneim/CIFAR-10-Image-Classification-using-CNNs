# CIFAR-10 Image Classification using CNNs

This repository contains a Convolutional Neural Network (CNN) implementation for classifying images from the CIFAR-10 dataset. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Overview:
1. **Dataset**: CIFAR-10 dataset is loaded and split into training and test sets. Pixel values are normalized to improve training efficiency.
   
2. **Data Augmentation**: ImageDataGenerator is used to apply augmentation techniques like rotation, shifting, and flipping to increase dataset variability, enhancing model generalization.

3. **Model Architecture**: 
   - Sequential CNN with multiple convolutional layers for feature extraction.
   - Batch Normalization layers to stabilize and accelerate training.
   - Dropout layers to prevent overfitting.
   - Dense layers for classification, with a softmax activation for outputting class probabilities.

4. **Training**: The model is compiled with Adam optimizer, categorical cross-entropy loss function, and accuracy metric. It is trained for 18 epochs with a batch size of 64.

5. **Evaluation**: The trained model achieves an accuracy of 80.85% on the test set, demonstrating its effectiveness in classifying CIFAR-10 images.

6. **Visualization**: 
   - Training and test accuracy/loss curves are plotted to visualize model performance over epochs.
   - Sample images from the test set are displayed with their actual and predicted labels, showcasing the model's predictions.

## Dependencies:
- TensorFlow 2.x
- Matplotlib

## Results:
The trained model achieves an accuracy of 80.85% on the test set, indicating good performance in classifying CIFAR-10 images.

This project demonstrates the application of CNNs for image classification, utilizing data augmentation, batch normalization, dropout, and other techniques to build an effective model for the CIFAR-10 dataset.
