import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# Load known faces and their names
known_faces = []
known_face_names = []

# Load known faces from images and add their names
def load_known_faces(image_paths):
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        name = path.split("/")[-1].split(".")[0]  # Extract the name from the file path
        known_face_names.append(name)

# Load known faces and their names
image_paths = ["known_faces/face1.jpg","known_faces/Monty.jpg","known_faces/charan.jpg","known_faces/sanjay.jpg","known_faces/hari.jpg","known_faces/avi.jpg","known_faces/saketh.jpg","Known_faces/pavan.jpg","Known_faces/Funny.jpg",]  # Add paths to your known faces
load_known_faces(image_paths)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize variables for attendance tracking
attendance = {name: False for name in known_face_names}

# Create a CSV file for attendancer̥
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{current_date}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Name', 'Time'])

while True:
    _, frame = video_capture.read()

    # Find all face locations and encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face encoding to known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Attentive"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Record attendance if the person is known and not already marked
        if name in attendance and not attendance[name]:
            attendance[name] = True
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([name, current_time])

        # Draw a rectangle and label for the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition Attendance', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
video_capture.release()
cv2.destroyAllWindows()