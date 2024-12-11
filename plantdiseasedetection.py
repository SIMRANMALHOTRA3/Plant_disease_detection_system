import warnings
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
warnings.filterwarnings("ignore", category=InsecureRequestWarning)
import certifi
import gdown

# Use certifi to provide SSL certificates
gdown.download(url, output, quiet=False, verify=certifi.where())

import os
import streamlit as st
import sys
import streamlit as st
st.write("Streamlit is using Python at:", sys.executable)

# Try importing cv2 and handle ImportError
try:
    import cv2
except ImportError as e:
    st.error("The OpenCV library (cv2) is not installed. Please install it using 'pip install opencv-python'.")
    st.stop()  # Stop execution if cv2 is not available

import gdown
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array

# Function to download the model from Google Drive
def download_model():
    try:
        url = 'https://drive.google.com/uc?export=download&id=1uvAp_I30bpXBt3nHQfQiow7MoGZXsJYx'
        output = 'plant_disease_model.h5'
        if not os.path.exists(output):  # Download only if the model file doesn't exist
            gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Error downloading model: {e}")

# Function to load the trained model
def load_trained_model():
    try:
        download_model()
        model = load_model("plant_disease_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess the input image
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (256, 256))  # Resize image to match model input
        img = img_to_array(img)  # Convert image to array
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize pixel values
        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Streamlit interface setup
st.title("Plant Disease Detection")
st.write("Upload a plant leaf image and get predictions for common diseases.")

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    image_path = "temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the trained model
    model = load_trained_model()
    
    if model:
        # Preprocess the uploaded image
        processed_img = preprocess_image(image_path)

        if processed_img is not None:
            try:
                # Make a prediction using the trained model
                prediction = model.predict(processed_img)
                classes = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
                predicted_class = classes[np.argmax(prediction)]

                st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
                st.write(f"Prediction: {predicted_class}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Error processing the image for prediction.")
    else:
        st.error("Model could not be loaded.")
else:
    st.write("Please upload an image to get started.")
