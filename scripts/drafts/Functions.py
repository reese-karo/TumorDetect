import tensorflow as tf
import numpy as np
import cv2
from transformers import pipeline
from openai import OpenAI


def preprocess_image(image, example = False):
    '''
    This function preprocesses the image for the model.
    - If img is an example, it will read in the image.
    - if img is not an example, it will skip the reading since its already an image.
    Args:
        image (numpy.ndarray): The image to preprocess.
        example (bool): Whether the image is from the example folder.

    Returns:
        numpy.ndarray: The preprocessed image.
    '''
    try:
        # if example, read in the image since its from the example folder
        if example:
            img = cv2.imread(image) # read in the image if its from the example folder
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # just convert the image to rgb
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
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a user on a tumor detection app that will provide a summary of the tumor type. You are limited to these four tumor types: glioma, meningioma, normal, pituitary."},
                {"role": "user", "content": f"Provide a clear and concise summary of the tumor {tumor_type}. Include a description of the tumor and some facts about it that are relevant"}
            ]
        )
        # Clean up and return the response
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the OpenAI API: {e}"
    
def predict_image(image, model, example = False):
    '''
    This function predicts the tumor type of the image using the cnn trained model.
    '''
    label_names = ['Glioma', 'Meningioma', 'Normal', 'Pituitary'] # set of the tumor types for the model
    try:
        processed_image = preprocess_image(image, example) # preprocess the image
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class] * 100  # Extract confidence score
        predicted_label = label_names[predicted_class]
        return f"{predicted_label} (CONFIDENCE: {confidence:.2f}%)"
    except Exception as e:
        return f"An error occurred with the image: {e}"