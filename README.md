# 🌿 Plant Disease Detection System

This project uses deep learning to classify plant leaf diseases from images. It’s designed to help automate the detection of plant diseases using Convolutional Neural Networks (CNNs). The system was trained on a labeled dataset of diseased and healthy leaf images, enabling it to predict the disease category based on visual symptoms.

---

## 📌 Project Goals

- Detect and classify diseases in plant leaves using image-based analysis
- Automate manual inspection to reduce time and error
- Build a reliable deep learning model with high accuracy
- Evaluate the model performance using standard metrics like accuracy, precision, and recall

---

## 🧰 Technologies Used

| Task                   | Tool / Library              |
|------------------------|-----------------------------|
| Language               | Python 3                    |
| Deep Learning          | TensorFlow, Keras           |
| Data Handling          | NumPy, Pandas               |
| Image Processing       | OpenCV, Pillow              |
| Model Evaluation       | scikit-learn                |
| Visualization          | Matplotlib                  |
| File Management        | gdown (for downloading models from Google Drive) |

---

## 🧠 Model Architecture

- A custom Convolutional Neural Network (CNN) built using Keras
- Includes convolutional, max pooling, dropout, and dense layers
- Trained on images resized to a fixed input shape (e.g., 128x128)
- Uses categorical crossentropy as the loss function with softmax activation for multi-class classification
- Optimizer: Adam

---

## 📂 Project Structure

plant-disease-detection/
├── dataset/ # Local dataset (if used)
├── model/ # Saved .h5 trained model
├── model_training.ipynb # Jupyter notebook for training
├── prediction_script.py # Script to make predictions on new images
├── utils.py # Image preprocessing helpers
├── requirements.txt
└── README.md

---

## 📦 Installation

Make sure you have Python 3.7 or later installed.


# Clone the repository
git clone https://github.com/SIMRANMALHOTRA3/Plant_disease_detection_system
cd plant-disease-detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt



## 🔍 How to Use

**1. Train the model:**

Open `model_training.ipynb` and follow the steps to:

- Load the dataset  
- Preprocess images  
- Build and compile the model  
- Train and evaluate the model  
- Save the model (.h5 file)  

**2. Predict on new images:**

Use the `prediction_script.py` to test the model:

python prediction_script.py --image path_to_leaf.jpg


## 🧪 Model Performance

- Validation Accuracy: ~95%  
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score  
- Confusion Matrix and training graphs are available in the notebook  

---

## 🌿 Sample Predictions

| Input Image    | Predicted Disease      | Confidence |
|----------------|-----------------------|------------|
| leaf_001.jpg   | Tomato Early Blight   | 96.2%      |
| leaf_045.jpg   | Potato Late Blight    | 94.5%      |
| leaf_107.jpg   | Healthy               | 98.8%      |

---


## 🔮 Future Enhancements

- Add a web interface (Streamlit or Flask)  
- Extend support to more plant species  
- Integrate real-time detection using a mobile camera  
- Deploy the model as an API  

Email: simranmalhotra3399@gmail.com
