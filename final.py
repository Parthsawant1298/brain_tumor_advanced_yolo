import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2
import os

# Constants
MODEL_PATH_CNN = "models/mri.keras"
MODEL_PATH_YOLO = "models/best_model (5).pt"
OUTPUT_IMAGE_PATH = "annotated_image.jpg"
CONFIDENCE_THRESHOLD = 0.5  # Confidence threshold for YOLO predictions

# Define class labels for CNN
class_labels = {0: 'Tumor present (glioma)', 1: 'Tumor present (meningioma)', 2: 'No tumor', 3: 'Tumor present (pituitary)'}

def load_cnn_model(model_path):
    """Load CNN model from the given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_yolo_model(model_path):
    """Load YOLO model from the given path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return YOLO(model_path)

def predict_tumor(img, cnn_model):
    """Predict tumor type using the CNN model."""
    img = img.convert('RGB')  # Ensure the image is in RGB format
    img = img.resize((64, 64))  # Resize to match training images
    img_array = np.array(img, dtype=np.float32)  # Convert image to numpy array with float32 type
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    result = cnn_model.predict(img_array)
    predicted_class_index = np.argmax(result[0])
    prediction = class_labels[predicted_class_index]

    return prediction, result[0]

def load_image(image_path):
    """Load an image from the given path using OpenCV."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image

def get_most_confident_box(results, threshold=CONFIDENCE_THRESHOLD):
    """Get the most confident bounding box."""
    most_confident_box = None
    highest_confidence = threshold
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0].item()
            if confidence >= highest_confidence:
                most_confident_box = box
                highest_confidence = confidence
    return most_confident_box

def annotate_image(image, box, dpi=96):
    """Annotate image with the most confident bounding box and its size in mm."""
    if box is not None:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get the bounding box coordinates
        score = float(box.conf[0].item())  # Convert tensor to float
        label = f'Confidence: {score * 100:.2f}%'

        # Convert pixel dimensions to mm
        px_to_mm = 25.4 / dpi
        width_mm = (x2 - x1) * px_to_mm
        height_mm = (y2 - y1) * px_to_mm
        size_label = f'{width_mm:.2f}mm x {height_mm:.2f}mm'

        # Draw the bounding box in red
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # Put the confidence label above the bounding box in red
        cv2.putText(image, label, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # Put the size label above the bounding box in red
        cv2.putText(image, size_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

def save_image(image, output_path):
    """Save the annotated image to the given path."""
    success = cv2.imwrite(output_path, image)
    if not success:
        raise IOError(f"Failed to save image at: {output_path}")

def main():
    st.title("MRI Image Analysis")

    # Load models
    cnn_model = load_cnn_model(MODEL_PATH_CNN)
    yolo_model = load_yolo_model(MODEL_PATH_YOLO)

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Predict tumor type
        img = Image.open(uploaded_file)
        prediction, probabilities = predict_tumor(img, cnn_model)
        st.write(f"Prediction: {prediction}")
        st.write(f"Prediction probabilities: {probabilities}")

        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        img.save(temp_image_path)

        # Perform object detection
        results = yolo_model.predict(source=temp_image_path)

        # Get the most confident bounding box
        most_confident_box = get_most_confident_box(results)

        # Load and annotate the image
        image = load_image(temp_image_path)
        annotate_image(image, most_confident_box)
        save_image(image, OUTPUT_IMAGE_PATH)

        # Display the annotated image
        st.image(OUTPUT_IMAGE_PATH, caption='Annotated MRI Image', use_column_width=True)

if __name__ == "__main__":
    main()
