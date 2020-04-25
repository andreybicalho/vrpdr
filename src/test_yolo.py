import cv2 as cv
import argparse
import sys
import os.path
from yolo import Yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing YOLO...')
    parser.add_argument('--image', help='Path to image file.')
    args = parser.parse_args()

    yolo = Yolo(img_width=1056, img_height=576, 
                debug=True, confidence_threshold=0.6, non_max_supress_theshold=0.4,
                classes_filename='../config/classes.names',
                model_architecture_filename="../config/yolov3_license_plates.cfg", 
                model_weights_filename="../config/yolov3_license_plates_last.weights",
                output_directory='../debug/')

    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)

    # get frame
    hasFrame, frame = cap.read()

    if hasFrame:
        yolo.detect(frame)
    else:
        print("Frame not found!")
