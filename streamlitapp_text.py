import datetime
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from transformers import pipeline

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the pneumonia detection model
pneumonia_model = load_model("./models/keras_model.h5", compile=False)

# Load the labels (assuming you might still need them for classification purposes)
class_names = open("./models/labels.txt", "r").readlines()

# Load text generation model
generator = pipeline('text-generation', model='gpt-2')

def preprocess_image(img):
    # Preprocess the image for the pneumonia model
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

def detect_pneumonia(img_array):
    predictions = pneumonia_model.predict(img_array)
    
    # Example interpretation of model output
    pneumonia_present = predictions[0][0] > 0.5  # Assuming binary classification
    severity = "Moderate" if pneumonia_present else "None"
    
    return {"pneumonia_present": pneumonia_present, "severity": severity}

def generate_report(pneumonia_results):
    # Generate a report
    report = f"""
    Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Pneumonia Detection:
    Pneumonia Present: {'Yes' if pneumonia_results['pneumonia_present'] else 'No'}
    Severity: {pneumonia_results['severity'] if pneumonia_results['pneumonia_present'] else 'N/A'}
    
    Recommendations:
    {'Further examination required' if pneumonia_results['pneumonia_present'] else 'No further action needed'}
    """

    # Optionally, generate additional details using text generation
    additional_info = generator("Provide additional clinical recommendations based on the following information: " + report, max_length=100)[0]['generated_text']
    report += f"\nAdditional Recommendations:\n{additional_info}"

    return report

# Streamlit interface
st.title("Pneumonia Detection and Reporting")
st.header("Upload an Image for Pneumonia Detection")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img_array = preprocess_image(image)
    
    # Perform pneumonia detection
    pneumonia_results = detect_pneumonia(img_array)

    # Generate and display the report
    report = generate_report(pneumonia_results)
    st.text_area("Automated Report", report, height=300)

    # Save the report to a file
    if st.button('Save Report'):
        with open('automated_pneumonia_report.txt', 'w') as file:
            file.write(report)
        st.success("Report saved successfully!")
