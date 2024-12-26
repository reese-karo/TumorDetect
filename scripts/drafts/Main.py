# this file will be used to run the front end where the user can upload their own image and get a result.
# from the result, we will use hugging face to allow the user to interact with the result and learn more about it.
# load in the necessary libraries
__author__ = "Reese Karo"
__version__ = "1.0.0"
__date__ = "2024-12-25"

import tensorflow as tf
import os
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from transformers import pipeline
import openai
# set the working directory
os.chdir("/Users/reese/Documents/Projects/BioStats/TumorDetect")

### functions for the app ###

label_names = ['glioma', 'meningioma', 'normal', 'pituitary'] # set of the tumor types for the model
image_model = load_model('scripts/transfer_learned_model.keras') # load in the cnn trained model

def preprocess_image(image):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError("Invalid image. Please upload a valid image file.") from e

text_generator = pipeline("text-generation", model="gpt2")

def generate_summary(tumor_type, api_key):
    '''
    Generates a summary of the tumor type using OpenAI's GPT model.
    '''
    try:
        openai.api_key = api_key,
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            prompt=f"Provide a clear and concise summary of the {tumor_type} tumor. Include its causes, symptoms, and treatment options.",
            max_tokens=200,
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        # Clean up and return the response
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"
    
def predict_image(image, api_key):
    '''
    This function predicts the tumor type of the image using the cnn trained model.
    '''
    processed_image = preprocess_image(image) # preprocess the image
    prediction = image_model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]  # Extract confidence score
    predicted_label = label_names[predicted_class]
    summary = generate_summary(predicted_label, api_key)
    return f"{predicted_label} (CONFIDENCE: {confidence:.2f}%)\n SUMMARY:\n{summary}"

### main app ###
st.set_page_config(layout="wide", page_title="Tumor Detection")
st.title("Tumor Detection")
st.write("Use this app to detect tumors from your MRI scans, and to receive a summary of the tumor.")
st.write("Upload an MRI scan to get started.")
uploaded_file = st.sidebar.file_uploader("Choose an MRI Scan to upload...", type=["jpg", "jpeg"])
api_key = st.sidebar.text_input("Enter your OPENAI API key:", type="password")
if uploaded_file is not None:
    try:
        image=cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image. Please upload a valid image file.")
        st.image(image, caption="Uploaded MRI Scan", use_column_width=False)

        with st.spinner("Analyzing MRI scan and generating prediction..."):
            result = predict_image(image)
        st.write(result)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an MRI scan to get started.")

        
with st.expander("About"):
    st.write("This app uses a pre-trained CNN model to analyze MRI scans and predict the type of tumor present. It also generates a summary of the tumor using a GPT-2 model from Hugging Face.")
    st.write("The model is trained on a dataset of MRI scans with labels for tumor type, and it uses transfer learning to apply this knowledge to new images.")
    st.write("The project was originally created by Will Mahnke, Reese Karo, Navin Lo, Casey, and Daniel Ledvin, I just wanted to make an app out of it:)")