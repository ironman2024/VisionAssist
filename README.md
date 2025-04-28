# VisionAssist 👁️ 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green)](https://github.com/ultralytics/yolov8)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<div align="center">
  <img src="Untitled video - Made with Clipchamp.mp4" alt="VisionAssist Demo">
</div>

VisionAssist is a groundbreaking AI-powered assistant that transforms the way visually impaired individuals interact with their environment. Using state-of-the-art computer vision and natural language processing, it provides real-time audio descriptions of surroundings, making the world more accessible and navigable.

## ✨ Features

- 🎯 **Real-time Object Detection**
  - Powered by YOLOv8, one of the fastest and most accurate object detection models
  - Detects 80+ different types of objects in real-time
  - Smooth performance on standard hardware

- 📏 **Precise Distance Estimation**
  - Accurate distance measurements using advanced focal length calculations
  - Real-time updates as objects move
  - Distance reported in both inches and feet

- 🔊 **Natural Audio Descriptions**
  - Crystal-clear text-to-speech descriptions
  - Contextual information about object locations
  - Adjustable speech rate and volume

- 🎤 **Intuitive Voice Control**
  - Simple voice commands for system control
  - Works in noisy environments
  - Supports multiple accents and dialects

## 🔧 Prerequisites

- Python 3.8 or higher
- Webcam or USB camera
- Microphone
- Internet connection (for speech recognition)
- 4GB RAM minimum (8GB recommended)
- NVIDIA GPU (optional, for better performance)

## ⚡ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/VisionAssist.git
    cd VisionAssist
    ```

2. **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the YOLO model:**
    ```bash
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    ```

## 🚀 Usage

1. **Start the application:**
    ```bash
    python main.py
    ```

2. **Voice Commands:**
   - Say "start" to begin object detection
   - Say "stop" to pause detection
   - Say "quit" to exit the application

## 🔍 How It Works

### Main Components

- **Object Detection and Distance Estimation**:
    ```python
    import cv2
    from ultralytics import YOLO
    
    Known_distance = 30  # Inches
    Known_width = 5.7  # Inches

    # Load the YOLO model
    model = YOLO('yolov8n.pt')  # Load an official model
    names = model.names  # Get class names
    ```

- **Text-to-Speech and Speech Recognition**:
    ```python
    import pyttsx3
    import speech_recognition as sr

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Initialize the speech recognizer
    recognizer = sr.Recognizer()
    ```

- **Focal Length Calculation**:
    ```python
    def FocalLength(measured_distance, real_width, width_in_rf_image):
        return (width_in_rf_image * measured_distance) / real_width
    ```

- **Distance Finder**:
    ```python
    def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
        return (real_object_width * Focal_Length) / object_width_in_frame
    ```

- **Generate Description**:
    ```python
    def generate_description(object_distance, class_id):
        return f"A {class_id} is at {object_distance} inches"
    ```

- **Generate Speech**:
    ```python
    def generate_speech(description):
        engine.say(description)
        engine.runAndWait()
        print("Say 'start' to continue or 'stop' to end.")
        command = listen_for_command()
        return command
    ```

- **Listen for Command**:
    ```python
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
    ```

- **Control Speech**:
    ```python
    def control_speech():
        while True:
            command = generate_speech("Description goes here")
            if command == "start":
                cap = cv2.VideoCapture(0)  # Camera object
                describe_objects(cap)
            elif command == "stop":
                print("Stopping speech generation...")
                engine.stop()
                break
            else:
                print("Sorry, could not understand the command.")
    ```

- **Describe Objects**:
    ```python
    def describe_objects(cap):
        Focal_length_found = None  # Initialize Focal_length_found variable
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)  # Predict on an image
            result = results[0]

            if Focal_length_found is None:
                Focal_length_found = FocalLength(Known_distance, Known_width, frame.shape[1])

            for box in result.boxes:
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                x, y, w, h = cords
                class_id = result.names[box.cls[0].item()]

                object_width_in_frame = w
                object_distance = Distance_finder(Focal_length_found, Known_width, object_width_in_frame)
                object_distance = round(object_distance, 2)

                description = generate_description(object_distance, class_id)
                command = generate_speech(description)
                if command == "stop":
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Object: {class_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    ```

### Main Function

- **Main Function**:
    ```python
    def main():
        control_speech()

    if __name__ == "__main__":
        main()
    ```

## ❗ Troubleshooting

Common issues and solutions:

1. **Camera not detected:**
   ```bash
   # Try changing the camera index
   cv2.VideoCapture(1)  # Instead of 0
   ```

2. **Speech recognition errors:**
   - Ensure stable internet connection
   - Check microphone permissions
   - Try reducing background noise

3. **Performance issues:**
   - Close other GPU-intensive applications
   - Reduce frame resolution in settings
   - Use a faster YOLO model variant

## 🤝 Contributing

We love your input! Check out our [Contributing Guidelines](CONTRIBUTING.md) to get started.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Support

If you find this project useful, please consider giving it a star ⭐️

## 📧 Contact

For any questions or support, please open an issue or contact us at [your-email@example.com](mailto:your-email@example.com)

---

<div align="center">
Made with ❤️ for the visually impaired community
</div>
