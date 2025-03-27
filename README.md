### 📝 **README: Handwritten Digit Recognition Using TensorFlow and MNIST Dataset**

---

## 📌 **Project Title:**  
🎯 **Handwritten Digit Recognition with TensorFlow Using MNIST Dataset**

---

## 🔥 **Project Description**

This project demonstrates the **Handwritten Digit Recognition System** using a **Deep Neural Network (DNN)** built with **TensorFlow**. It uses the **MNIST dataset**, which contains handwritten digit images (0-9), and classifies them with high accuracy.  

The model:  
- Uses a **Deep Neural Network (DNN)** with multiple hidden layers.  
- Applies **ReLU activation** for non-linearity.  
- Uses **Softmax activation** in the output layer for multi-class classification.  
- Employs **Adam optimizer** for efficient weight updates.  
- Evaluates the model with **accuracy, confusion matrix, and classification report**.  
- Visualizes **training accuracy, loss, and sample predictions**.  

---

## 🚀 **Features and Functionality**

✅ **Real-World Dataset:**  
- Utilizes the **MNIST dataset** containing 60,000 training and 10,000 test images.  
- Each image is **28x28 pixels** and represents a digit from `0-9`.  

✅ **Deep Learning Model:**  
- **Deep Neural Network (DNN)** with:  
    - 3 hidden layers (256, 128, and 64 neurons).  
    - **ReLU activation** for each hidden layer.  
    - **Softmax activation** for multi-class classification.  
- **Adam optimizer** for efficient training.  
- **Cross-entropy loss** as the cost function.  

✅ **Model Evaluation:**  
- **Accuracy:** Measures the model's correctness.  
- **Confusion Matrix:** Displays classification results.  
- **Precision, Recall, and F1-Score:** For detailed performance analysis.  

✅ **Visualization:**  
- **Loss and Accuracy Curves:** Shows model performance over epochs.  
- **Sample Predictions:** Displays the model's predictions on real test images.  

---

## 📊 **Technologies Used**

- **Python:** Core programming language.  
- **TensorFlow:** Deep learning framework for model building.  
- **NumPy:** For numerical operations.  
- **Matplotlib:** For visualizing accuracy, loss, and predictions.  
- **Scikit-Learn:** For generating the confusion matrix and classification report.  

---

## 📚 **Dataset Overview**

The **MNIST dataset** contains:  
- **60,000 training images**.  
- **10,000 test images**.  
- Each image is grayscale with **28x28 pixels**.  
- **Labels:**  
    - `0` → Digit zero  
    - `1` → Digit one  
    - `...`  
    - `9` → Digit nine  

---

## 💡 **Model Architecture**

### ✅ **Data Preparation:**
1. **Load the MNIST dataset.**  
2. **Normalize the pixel values** between `0` and `1`.  
3. **Reshape the images** into a flattened format:  
    - `28x28` → `784` features.  
4. **One-hot encode the labels** into 10 output classes.

---

### ✅ **Deep Neural Network (DNN) Model:**

1. **Input Layer:**  
   - **784 neurons** (28x28 pixel image flattened).  
2. **Hidden Layers:**  
   - **256 neurons** → `ReLU activation`.  
   - **128 neurons** → `ReLU activation`.  
   - **64 neurons** → `ReLU activation`.  
3. **Output Layer:**  
   - **10 neurons** → `Softmax activation` for multi-class classification.  
4. **Optimizer:**  
   - **Adam optimizer** for efficient training.  
5. **Loss Function:**  
   - **Categorical cross-entropy loss** for multi-class classification.  

---

## 📈 **Model Performance and Results**

✅ **Training and Validation Accuracy:**  
```
Training Accuracy: 98.5%  
Validation Accuracy: 97.4%  
```

✅ **Test Accuracy:**  
```
Test Accuracy: 97.1%  
```

✅ **Precision, Recall, and F1-Score:**  
```
Precision: 0.97  
Recall: 0.97  
F1-Score: 0.97  
```

---

## 👨‍🏫 **Credits**

This project is inspired by the course:  
📚 **"Advanced Learning Algorithms" by Andrew Ng** on **Coursera**.  
Link: [Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms)  
