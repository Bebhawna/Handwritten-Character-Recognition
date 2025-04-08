# 🧠 Hindi Character Recognition using CNN

This project aims to build a Convolutional Neural Network (CNN) model to classify handwritten Devanagari (Hindi) characters and digits using the **Devanagari Handwritten Character Dataset**.

## 📦 Dataset

- Dataset Source: [Kaggle - Hindi Character Recognition](https://www.kaggle.com/datasets/suvooo/hindi-character-recognition)
- The dataset contains 46 classes (36 characters + 10 digits), split into `Train` and `Test` folders.

## 🚀 Libraries Used

- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- OpenCV
- Scikit-learn

## 📊 Model Architecture

- 3 Convolutional Layers with ReLU activation
- Batch Normalization after each Conv layer
- MaxPooling Layers
- Flatten + Dense Layers
- Dropout Regularization
- Output layer with Softmax for 46 classes

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(46, activation='softmax')
])
```
## Installation
Clone the repositpory: 
```bash
git clone https://github.com/bebhawna/handwritten-character-recognition.git
cd handwritten-character-recognition
```
Install dependencies

```bash
pip install -r requirements.txt
```
## Running the project
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Load your saved model
model = load_model('model/character_recognition_model.h5')

# Load and preprocess the image
test_img = cv2.imread('path_to_your_image.jpg')  # 👈 Replace with your own image path
test_img = cv2.resize(test_img, (64, 64))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # Optional: if your model needs grayscale
test_img = test_img / 255.0  # Normalize pixel values
test_img = test_img.reshape(1, 64, 64, 1)  # Add batch and channel dimensions

# Predict
prediction = model.predict(test_img)
predicted_class = np.argmax(prediction)

# Display the image and prediction
plt.imshow(cv2.imread('path_to_your_image.jpg'))
plt.title(f'Predicted Class: {predicted_class}')
plt.axis('off')
plt.show()

Make sure your image is:
1.Clear and centered handwritten character
2.In JPEG/PNG format
3.Preferably black on white background

## Contributing:
Contributions are welcome!
If you have suggestions or improvements, feel free to open an issue or submit a pull request.

✍️ Author
Bhawna Bisht


