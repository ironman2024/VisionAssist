# VisionAssist

VisionAssist is a project designed to assist blind people by providing audio descriptions of their surroundings. The system uses computer vision and natural language processing to detect objects and estimate their distance from the user, then generates spoken descriptions to inform the user about their environment.

## Features

- **Object Detection**: Uses YOLO (You Only Look Once) model to detect objects in real-time.
- **Distance Estimation**: Estimates the distance of detected objects from the user.
- **Audio Descriptions**: Generates and speaks out descriptions of the detected objects.
- **Voice Control**: Allows users to control the system using voice commands.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/VisionAssist.git
    cd VisionAssist
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the YOLO model file `yolov8n.pt` in the project directory. You can download it from the official YOLO repository or use a custom model.

## Usage

1. Run the main script:
    ```bash
    python main.py
    ```

2. The system will start and prompt you to say "start" to begin object detection or "stop" to terminate the process.

## Code Overview

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

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have feature requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
