# this file will be used to run the front end where the user can upload their own image and get a result.
# from the result, we will use hugging face to allow the user to interact with the result and learn more about it.
import tensorflow as tf
import os
import numpy as np
import cv2
import gradio as gr
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

os.chdir("/Users/reese/Documents/Projects/BioStats/TumorDetect")
print(os.getcwd())

###### this section is for loading in the labels so that we can use them to predict the user's image ######

label_names = ['glioma', 'meningioma', 'pituitary', 'normal']

# load in the cnn trained model
image_model = load_model('scripts/transfer_learned_model.keras') # this is the model that we will use to predict the user's image

def preprocess_image(image):
    try:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError("Invalid image. Please upload a valid image file.") from e

def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = image_model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]  # Extract confidence score
    predicted_label = label_names[predicted_class]
    return f"{predicted_label} (Confidence: {confidence:.2f})"


iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(width=224, height=224),
    outputs=gr.Label(label="Prediction"),
    examples=["data/glioma_tumor/G_1.jpg"],
    description="Upload an MRI scan to predict the type of brain tumor. Supported types: glioma, meningioma, pituitary, or normal."
)

iface.launch()
