# Machine Learning & Computer Vision Projects

This repository contains a collection of my Machine Learning, Computer Vision, and Applied Deep Learning projects completed across academic coursework, self-study, and personal experimentation. 

---
## 📂 1. Machine Learning Algorithms

Foundational ML algorithms implemented from scratch for learning and experimentation.

### **KNN**
- Distance metrics and k-value effects.
- Visualization of classification boundaries.

### **Kernels**
- Basic kernel similarity computations.

### **Naive Bayes**
- Gaussian and Multinomial variants.
- Notebook includes toy datasets and intuition.

### **Neural Networks**
- Simple feed-forward networks with manual backpropagation.
- Small-scale experiments for understanding learning dynamics.

### **Perceptron**
- Binary classification with perceptron learning rule.
- Includes visual demonstrations of linear separability.

---
## 📂 2. Computer Vision

### **NeRF (Neural Radiance Fields)**
- Implementation of NeRF for novel view synthesis.
- Covers positional encodings, ray sampling, and rendering.
- Demonstrates reconstruction of new viewpoints.

### **ResNet & UNet**
- Training deep CNNs for classification and segmentation tasks.
- Includes:
  - ResNet, UNet and Binary Segmentation

---
## 📂 3. Research Project: Kernel Stability & Numerical Methods

Advanced numerical experiments exploring kernel behavior, conditioning, and stable differencing—part of a research study on Gaussian Processes and kernel methods.

### **Experiments with Clustered Points and Preconditioning**
- Studies ill-conditioning of kernel matrices when data points cluster.
- Implements and analyzes preconditioning strategies.
- Visualizes conditioning numbers for different configurations.

### **IMQ Kernels: Stable Differencing Using mpmath**
- High-precision numerical experiments using `mpmath`.
- Implements stable differencing formulas for IMQ kernels.
- Demonstrates how small perturbations and spacing affect kernel stability.
- Includes plots and comparisons of naive vs stable formulations.


---
## 📂 4. Cryptography + CNN

### **Image-Based Secret Sharing with CNN Enhancement**
`cnn_with_rg_with_validation.ipynb`

- Implements **random-grid secret sharing** using OpenCV and NumPy.
- Trains a **CNN on MNIST** (TensorFlow/Keras) to enhance reconstruction of encrypted shares.
- Includes:
  - Preprocessing pipeline
  - Model training + validation
  - Comparison of reconstruction quality with/without CNN assistance
