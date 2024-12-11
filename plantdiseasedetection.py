import os
import gdown
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import img_to_array
import certifi  # For SSL certificate verification

def download_model():
    try:
        url = 'https://drive.google.com/uc?export=download&id=1uvAp_I30bpXBt3nHQfQiow7MoGZXsJYx'
        output = 'plant_disease_model.h5'
        if not os.path.exists(output):  # Download only if the model file doesn't exist
            gdown.download(url, output, quiet=False, verify=certifi.where())  # Use certifi for SSL verification
    except Exception as e:
        st.error(f"Error downloading model: {e}")

def load_trained_model():
    try:
        download_model()
        model = load_model("plant_disease_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess and predict logic follows...
