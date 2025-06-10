# AI Image Classifier

This project is an AI-based image classifier that determines whether a given image is AI-generated or real. The model was trained on a dataset of 50,000 images to achieve accurate classification performance.

## Live Demo

You can try the model live at:
**[https://ai-image-classifier.streamlit.app/](https://ai-image-classifier.streamlit.app/)**

## Running Locally

To run the application on your local machine:

1. Install the required libraries listed in `requirements.txt`.
2. Ensure that the Python script (`app.py`) and the trained model file are in the same directory.
3. Start the application using the following command:

   ```
   streamlit run app.py
   ```

## Frameworks and Tools Used

* **Selenium**: For web scraping and data collection
* **TensorFlow**: For training and deploying the classification model
* **Streamlit**: For building and hosting the web interface

## Directory Structure

```
project/
│
├── app.py                 # Streamlit web app
├── model.h5               # Trained TensorFlow model
├── requirements.txt       # Python dependencies
└── ...
```

---
