import cv2
import numpy as np
import os

# Load the pre-trained MobileNet SSD model and configuration
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define the classes in the MobileNet SSD
class_names = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 
               5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 
               10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 
               14: 'motorbike', 15: 'person', 16: 'pottedplant', 
               17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

def detect_objects(input_source=0):
    # Open the video capture (file or webcam)
    cap = cv2.VideoCapture(input_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare the frame for the DNN model
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Process the detections
        (h, w) = frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:  # You can adjust the threshold
                # Get the class ID and bounding box coordinates
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box and label on the frame
                label = f"{class_names[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow("Object Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define the path to the video (or None to use the webcam)
video_path = 'video.mp4'  # Set your video file path here

# Check if the video path is valid and exists, else use the webcam (0)
if os.path.exists(video_path):
    detect_objects(video_path)  # Open video file
else:
    detect_objects(0)  # Open webcam (0 is the default camera)
