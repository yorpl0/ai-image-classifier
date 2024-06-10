


model=load_model('workingclassifier.h5')
def predict(fixed):
    
    resize=tf.image.resize(fixed,(256,256))
    yhat = model.predict(np.expand_dims(resize/255, 0))
    
    
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='centered-title'>Image Classification with TensorFlow</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='centered-text'>Upload your image for classification</h2>",unsafe_allow_html=True)
uploaded_img=st.file_uploader("Upload here",type=["jpg","png","jpeg"])
if uploaded_img is not None:
    img = Image.open(uploaded_img)
    #print the image
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    #convert into arrays for opencv
    img = np.array(img)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    prediction = predict(image)
    st.write(f'the probability that the image is real:{prediction}')
    




