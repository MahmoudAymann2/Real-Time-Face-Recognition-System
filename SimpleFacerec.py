import cv2
import face_recognition
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25  # Resize frame to process faster

    def load_encoding_images(self, images_path):
        # Load all images from the directory
        images = os.listdir(images_path)
        for img_name in images:
            img_path = os.path.join(images_path, img_name)
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Encode faces (check if faces exist in the image)
            face_encodings = face_recognition.face_encodings(rgb_img)
            if len(face_encodings) > 0:
                img_encoding = face_encodings[0]  # Use the first face encoding
                self.known_face_encodings.append(img_encoding)
                self.known_face_names.append(os.path.splitext(img_name)[0])
                print(f"Loaded encoding for {img_name}")
            else:
                print(f"No face found in {img_name}, skipping...")

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Check if the detected face matches known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Get the closest known face if there is a match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back up to the original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        face_locations = face_locations.astype(int)

        return face_locations, face_names


# Usage example
if __name__ == "__main__":
    # Create an instance of SimpleFacerec
    sfr = SimpleFacerec()

    # Load encoded images
    sfr.load_encoding_images("C:\\Users\\Mandoo\\Pictures\\lock")

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect known faces in the frame
        face_locations, face_names = sfr.detect_known_faces(frame)

        # Display results
        for face_loc, name in zip(face_locations, face_names):
            # Draw rectangle around the face
            top, right, bottom, left = face_loc
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Display the name of the person
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Show the video frame with face detection
        cv2.imshow("Frame", frame)

        # Break the loop if 'Esc' is pressed
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
