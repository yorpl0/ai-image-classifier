import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.api.models import load_model
import numpy as np

# Load the model
model = load_model('v1.5.h5')

def predict(fixed):
    if fixed.ndim == 2:  # Grayscale image
        fixed = np.stack((fixed,)*3, axis=-1)
    elif fixed.ndim == 3 and fixed.shape[-1] == 4:  # RGBA image
        fixed = fixed[..., :3]
    
    resize = tf.image.resize(fixed, (256, 256))
    resize = tf.expand_dims(resize, axis=0)  # Add batch dimension
    resize = resize / 255.0  # Normalize the image
    yhat = model.predict(resize)
    return yhat[0][0]

# Custom CSS for background image and text styles
page_bg_img = '''
<style>
.stApp {
    background-image:url("https://images.unsplash.com/photo-1483366774565-c783b9f70e2c?fm=jpg&w=3000&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed; /* Fixed background image */
}

.centered-title {
    text-align: center;
    font-size: 2.5em;
    margin-top: -30px;
    font-family: 'Helvetica Neue', sans-serif;
    color: black; /* Text color */
    text-shadow: 2px 2px 4px white; /* White text shadow */
}

.centered-text {
    text-align: center;
    font-size: 1.2em;
    font-family: 'Helvetica Neue', sans-serif;
    color: black; /* Text color */
    text-shadow: 2px 2px 4px white; /* White text shadow */
}

.note {
    font-size: 0.9em;
    color: gray;
    text-align: center;
    margin-top: 20px;
    font-family: 'Helvetica Neue', sans-serif;
}

.uploaded-image {
    text-align: center;
}

.prediction {
    font-size: 1.5em;
    font-weight: bold;
    text-align: center;
    margin-top: 20px;
    font-family: 'Helvetica Neue', sans-serif;
    color: black; /* Text color */
    text-shadow: 2px 2px 4px white; /* White text shadow */
}

.file-upload-btn label {
    color: black !important; /* Text color for file upload button */
}
</style>
'''

# Page configuration
st.set_page_config(page_title="Image Classification with TensorFlow", layout="centered", initial_sidebar_state="collapsed")
st.markdown(page_bg_img, unsafe_allow_html=True)

# App title and instructions
st.markdown("<h1 class='centered-title'>Image Classification with TensorFlow</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>Upload your image for classification</h2>", unsafe_allow_html=True)

# Upload image
uploaded_img = st.file_uploader("Upload here", type=["jpg", "png", "jpeg"], key="fileUploader")
st.markdown("<p class='note'>Note: The model performs well on non-human figures such as animals, art, etc.</p>", unsafe_allow_html=True)

# Processing the uploaded image
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    
    # Display uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True, output_format='auto')
    
    # Process image and show prediction
    st.write("Classifying...")
    img = np.array(img)
    prediction = predict(img)
    st.markdown(f"<p class='prediction'>The probability that the image is real: {prediction:.2f}</p>", unsafe_allow_html=True)
