import os
import face_recognition
import cv2
import numpy as np

from utils.app_utils import find_images
from utils.app_utils import save_frame
from utils.app_utils import create_dir
from utils.console_utils import cls

# Constants for app
camera_port = 0
images_directory = "images"
path = os.path.abspath(os.path.dirname(__file__)) + "\\" + images_directory


def find_faces():
    files = find_images(path)

    # Create array of images
    images = []

    # Create arrays of known face encodings and their names
    known_face_encodings = []
    known_face_names = []

    for file in files:
        images.append(face_recognition.load_image_file(path + '\\' + file))
        known_face_names.append(file.__str__().replace(".jpg", ""))

    for image, image_number in zip(images, range(len(images))):
        try:
            # Returns recognized face as first element of an array
            known_face_encodings.append(face_recognition.face_encodings(image)[0])
        except IndexError:
            print(f"Can't find face on image {known_face_names[image_number]}")

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    video_capture = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    while True:
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If more than one face is matched with face encodings,
                # uses face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 200), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 200), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    # Creating dir if not exist
    create_dir(path)

    while True:
        res = int(input("1 - start face searching\n"
                        "2 - add new face to database\n"))
        if res == 1:
            find_faces()
        elif res == 2:
            save_frame(camera_port, images_directory)
        else:
            break

        cls()


if __name__ == "__main__":
    main()
