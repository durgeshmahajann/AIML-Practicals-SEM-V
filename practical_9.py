# AIML Practical 9
# Apply Object Detection from CCTV Footage

import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open CCTV video
video = cv2.VideoCapture("cctv.mp4")

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detection results
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("CCTV Object Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video.release()
cv2.destroyAllWindows()