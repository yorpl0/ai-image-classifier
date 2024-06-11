import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.api.models import load_model
import numpy as np
model=load_model('v1.5.h5')
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
st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size:30px;
        margin-top:-50px;
    }
    .centered-text {
        text-align: center;
    }
    .note {
        font-size: 12px;
        color: gray;
        text-align: center;
    }
    </style>""",unsafe_allow_html=True,
)
gray_text= """

        <p class="note">
        Note:Model is good at non-human figures such as animals,art,etc.
        </p>
"""

st.markdown("<h1 class='centered-title'>Image Classification with TensorFlow</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>Upload your image for classification</h2>",unsafe_allow_html=True)
uploaded_img = st.file_uploader("Upload here", type=["jpg", "png", "jpeg"])
st.markdown(gray_text,unsafe_allow_html=True)
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    # Print the image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert into arrays for TensorFlow
    img = np.array(img)
    prediction = predict(img)
    st.write(f'The probability that the image is real: {prediction}')




