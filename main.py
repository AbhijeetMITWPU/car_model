import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ✅ Load the trained model
MODEL_PATH = "best_model.h5"  # Update the path if needed
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Function to detect number plate in an image
def detect_plate(image):
    h, w, _ = image.shape
    input_image = cv2.resize(image, (224, 224)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Predict bounding box
    y_pred = model.predict(input_image, verbose=0)
    ymin, xmin, ymax, xmax = np.multiply(y_pred[0], [h, w, h, w])
    
    # Draw bounding box
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)
    return image

# ✅ Streamlit UI Setup
st.title("🚗 Car Number Plate Detection")
option = st.radio("Choose an option:", ("Upload Image", "Open Camera"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        result = detect_plate(image.copy())
        st.image(result, caption="Detected Plate", use_column_width=True)

elif option == "Open Camera":
    st.write("🔴 Press 'q' to stop the camera.")
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_plate(frame)
        st.image(frame, channels="BGR")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    st.success("Camera Stopped.")
