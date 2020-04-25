import cv2 as cv
import numpy as np
import argparse
import sys
import os.path
import logging
import matplotlib.pyplot as plt

from image_preprocessing import display_images, extract_chars
from ocr import OCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing OCR.')
    parser.add_argument('--image', help='Path to image file.')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    # Open the image file
    if not os.path.isfile(args.image):
        logging.debug("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)

    hasFrame, frame = cap.read()

    if hasFrame:
        img, characteres = extract_chars(frame, prefix_label='test_ocr', min_countours_area_ration=0.01, debug=True)

        #characteres = [img / 255 for img in characteres]
        #display_images(characteres, 3, 3)

        ocr = OCR(model_filename="../config/emnist_model.pt", use_cuda=False, debug=True)
        pred = ocr.predict(characteres)
        logging.info(f'\nPrediction: {pred}')
        
    else:
        logging.debug("Frame not found!")