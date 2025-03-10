import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Set page title and configuration
st.set_page_config(page_title="Car Number Plate Detection", layout="wide")

# ✅ Load the trained model
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ✅ Function to detect number plate in an image
def detect_plate(image):
    if image is None:
        return None
        
    h, w, _ = image.shape
    input_image = cv2.resize(image, (224, 224)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Get model
    model = load_model("best_model.h5")
    if model is None:
        return image
    
    # Predict bounding box
    y_pred = model.predict(input_image, verbose=0)
    ymin, xmin, ymax, xmax = np.multiply(y_pred[0], [h, w, h, w])
    
    # Draw bounding box
    result = image.copy()
    cv2.rectangle(result, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
    
    # Extract the plate region
    plate_region = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    return result, plate_region if plate_region.size > 0 else None

# UI Elements
st.title("🚗 Car Number Plate Detection")
st.write("Upload an image of a car to detect the number plate")

# Sidebar for options
with st.sidebar:
    st.header("Options")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    st.write("---")
    st.write("### About")
    st.write("This app uses a TensorFlow model to detect car number plates in images.")

# Main content
tab1, tab2 = st.tabs(["Upload Image", "Camera Input"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        image = np.array(Image.open(uploaded_file))
        
        # Display original image
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Detect Number Plate"):
            with st.spinner("Processing..."):
                # Detect plate
                try:
                    result, plate_region = detect_plate(image.copy())
                    
                    # Display results
                    st.subheader("Detection Result")
                    st.image(result, caption="Detected Plate", use_column_width=True)
                    
                    if plate_region is not None and plate_region.size > 0:
                        st.subheader("Extracted Plate")
                        st.image(plate_region, caption="Number Plate Region", use_column_width=True)
                    else:
                        st.warning("No plate region was found or it was too small.")
                        
                except Exception as e:
                    st.error(f"Error during detection: {e}")

with tab2:
    st.write("#### Camera Input")
    st.write("Note: Camera functionality in Streamlit web apps is limited. For best results, capture a photo and upload it.")
    
    # Streamlit's camera input
    camera_image = st.camera_input("Take a picture")
    
    if camera_image is not None:
        # Convert camera image to numpy array
        image = np.array(Image.open(camera_image))
        
        # Detect plate
        with st.spinner("Processing..."):
            try:
                result, plate_region = detect_plate(image.copy())
                
                # Display results
                st.subheader("Detection Result")
                st.image(result, caption="Detected Plate", use_column_width=True)
                
                if plate_region is not None and plate_region.size > 0:
                    st.subheader("Extracted Plate")
                    st.image(plate_region, caption="Number Plate Region", use_column_width=True)
                else:
                    st.warning("No plate region was found or it was too small.")
                    
            except Exception as e:
                st.error(f"Error during detection: {e}")

# Add footer
st.markdown("---")
st.markdown("Car Number Plate Detection App | Created with Streamlit and TensorFlow")
