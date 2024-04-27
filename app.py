import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.keras.models.load_model('C:/Users/X1 CARBON/Desktop/GlaucomaDetectionProject/models/glaucoma_model.h5')

def load_and_preprocess_image(image):
    image = Image.open(image).convert('RGB')
    image = image.resize((256, 256))  # Adjust the size to match your model's expected input
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Create a batch
    return image_array

def predict(image):
    image_array = load_and_preprocess_image(image)
    prediction = model.predict(image_array)
    return "Glaucoma Detected" if prediction[0][0] > 0.5 else "No Glaucoma"

st.title('Glaucoma Detection App')
st.write("Upload an image of an eye to detect glaucoma.")

uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_file)
    st.write(label)
