import os
from os import listdir
from os.path import isfile, join
import cv2


# Find all files in images/ directory
def find_images(dir_path):
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    return files


# Save frame to the file
def save_frame(camera_port, images_dir):
    video = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)

    while True:
        check, frame = video.read()
        cv2.imshow("Frame", frame)
        # Hit 's' on the keyboard to save image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            name = input("Write name for image: ")
            cv2.imwrite(f"{images_dir}/{name}.jpg", frame)
            break

    video.release()
    cv2.destroyAllWindows()

# Create directory
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
