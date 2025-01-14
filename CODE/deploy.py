import gradio as gr
import torch
from ultralytics import YOLO
import cv2


import numpy as np
import traceback
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = r"best.pt"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_image(image, target_size=(640, 640)):
    """
    Preprocess the image to match YOLO model's expected input
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target size for resizing
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    try:
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w = int(w * scale)
        new_h = int(h * scale)        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)        
        canvas = np.full((target_size[1], target_size[0], 3), 114, dtype=np.uint8)        
        start_x = (target_size[0] - new_w) // 2
        start_y = (target_size[1] - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return canvas
    except Exception as e:
        logger.error(f"Error in image preprocessing: {e}")
        logger.error(traceback.format_exc())
        return None

def load_model():
    """
    Load the YOLO model with error handling
    
    Returns:
        YOLO model or None if loading fails
    """
    try:
        logger.info(f"Attempting to load model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        
        logger.info(f"Model loaded. Checking model details:")
        logger.info(f"Number of classes: {model.model.nc if hasattr(model.model, 'nc') else 'Unknown'}")
        logger.info(f"Class names: {model.names if hasattr(model, 'names') else 'Unknown'}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

model = load_model()

def predict_image(input_image):
    """
    Perform object detection on the input image
    
    Args:
        input_image (numpy.ndarray): Input image for prediction
    
    Returns:
        numpy.ndarray: Image with detected objects annotated
    """
    if model is None:
        logger.error("Model is not loaded. Cannot make predictions.")
        return None
    
    if input_image is None:
        logger.warning("Input image is None")
        return None
    
    try:
        logger.info(f"Original image shape: {input_image.shape}")
        
        processed_image = preprocess_image(input_image)
        
        if processed_image is None:
            logger.error("Image preprocessing failed")
            return None
        
        logger.info(f"Processed image shape: {processed_image.shape}")
        
        results = model(processed_image, conf=0.1, iou=0.5)
        
        logger.info(f"Number of objects detected: {len(results[0].boxes)}")
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                logger.info(f"Detected: Class {box.cls}, Confidence: {box.conf}")
        
        annotated_image = results[0].plot()
        
        return annotated_image
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        return None

def main():
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(type="numpy", label="Upload Image"),
        outputs=gr.Image(type="numpy", label="Detected Objects"),
        title="YOLO Object Detection",
        description="Upload an image to detect objects using the trained YOLO model."
    )
    
    iface.launch(share=True)

if __name__ == "__main__":
    main()

print("Gradio app is ready to be launched!")