import streamlit as st

import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
import cv2

# Load the trained model
model = load_model('mbv2model1_split')

# Define the pest classes
pest_classes = ["Bollworms", "Fallarmyworms", "Thrips", "aphids", "black cutworms"]

# Function to preprocess an image
def preprocess(frame):
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame

# Function to predict pests in an image
def predict_pests(frame):
    processed_frame = preprocess(frame)
    prediction = model.predict(processed_frame)
    pests_detected = []
    for i, pred in enumerate(prediction[0]):
        if pred >= 0.9:  # Adjust threshold as needed
            pests_detected.append(f"Pest {pest_classes[i]}: {pred:.2f}")
    return pests_detected

# Title and image uploader
st.title("Pest Detection App")
uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Process the image and display the prediction
    pests = predict_pests(image)
    st.image(image,caption="processed Image", use_column_width=True)
    st.write("Detected Pests:")
    if pests:
        for pest in pests:
            st.write(pest)
    else:
        st.write("No pests detected.")
