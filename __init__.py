import cv2
import pyttsx3
import speech_recognition as sr
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
from torch.serialization import add_safe_globals
from pathlib import Path
import os
import time  # Add time module



# Simplified safe globals configuration
required_classes = [
    DetectionModel,
    nn.Module,
    Conv,
    C2f,
    SPPF,
    Detect
]

# Configure torch settings
add_safe_globals(required_classes)
torch.backends.cudnn.enabled = False

# Constants
Known_distance = 30  # Inches
Known_width = 5.7  # Inches

# Initialize engines
engine = pyttsx3.init()
recognizer = sr.Recognizer()

def load_yolo_model():
    """Load YOLO model with proper configurations"""
    try:
        import torch.serialization
        torch.serialization.weights_only = False
        # Use direct model loading from ultralytics
        print("Loading YOLO model...")
        model = YOLO("yolov8n.pt")  # This will download if needed
        
        # Force CPU inference
        model.to('cpu')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize model
print("Initializing YOLO model...")
model = load_yolo_model()
if model is None:
    raise RuntimeError("Failed to load YOLO model")

names = model.names

# Function to calculate focal length
def FocalLength(measured_distance, real_width, width_in_rf_image):
    return (width_in_rf_image * measured_distance) / real_width


# Function to estimate distance
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    return (real_object_width * Focal_Length) / object_width_in_frame


# Function to generate a description of the frame
def generate_description(object_distance, class_id):
    return f"A {class_id} is at {object_distance} inches"

# Function for speech generation
def generate_speech(description):
    # Speak out the description
    engine.say(description)
    engine.runAndWait()

    # Ask for start/stop command
    print("Say 'start' to continue or 'stop' to end.")
    command = listen_for_command()
    return command


# Function to listen for start/stop command
def listen_for_command():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print("Received command:", command)
        return command
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return ""
    except sr.RequestError:
        print("Could not request results; check your internet connection.")
        return ""


# Function to control speech generation based on user commands
def get_object_info(box, Focal_length_found):
    """Helper function to consistently calculate object dimensions and distance"""
    cords = box.xyxy[0].tolist()
    x1, y1, x2, y2 = [round(x) for x in cords]
    width = x2 - x1  # Calculate width from coordinates
    height = y2 - y1
    
    # Calculate distance
    object_distance = Distance_finder(Focal_length_found, Known_width, width)
    return x1, y1, width, height, round(object_distance, 2)

def control_speech():
    cap = cv2.VideoCapture(0)  # Initialize camera
    running = True
    speaking_enabled = False
    last_speech_time = 0
    Focal_length_found = None
    
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate focal length once
        if Focal_length_found is None:
            Focal_length_found = FocalLength(Known_distance, Known_width, frame.shape[1])
            
        results = model(frame)
        result = results[0]
        
        # Draw detections
        for box in result.boxes:
            # Get consistent measurements
            x, y, w, h, distance = get_object_info(box, Focal_length_found)
            class_id = result.names[box.cls[0].item()]
            
            # Draw rectangle and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_id} ({distance} inches)", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('Object Detection', frame)
        
        # Check for keyboard commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
        # Handle voice commands and descriptions
        if not speaking_enabled:
            try:
                with sr.Microphone() as source:
                    print("Say 'start' to begin descriptions or 'stop' to end")
                    audio = recognizer.listen(source, timeout=1)
                    command = recognizer.recognize_google(audio).lower()
                    if command == "start":
                        speaking_enabled = True
                        last_speech_time = time.time()
                    elif command == "stop":
                        running = False
            except (sr.WaitTimeoutError, sr.UnknownValueError):
                pass
            except Exception as e:
                print(f"Error: {e}")
                
        # Generate descriptions with distance every 5 seconds if enabled
        if speaking_enabled and time.time() - last_speech_time >= 5:
            descriptions = []
            for box in result.boxes:
                class_id = result.names[box.cls[0].item()]
                # Use the same helper function for consistency
                _, _, _, _, distance = get_object_info(box, Focal_length_found)
                descriptions.append(f"I see a {class_id} at {distance} inches")
                
            if descriptions:
                description = ". ".join(descriptions)
                engine.say(description)
                engine.runAndWait()
            last_speech_time = time.time()
    
    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    control_speech()  # Just call control_speech directly


if __name__ == "__main__":
    main()
