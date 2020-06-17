import cv2 as cv
import numpy as np
import argparse
import sys
import os.path
import logging
import matplotlib.pyplot as plt

from image_processing import plot_images, extract_chars, cv_skeletonize
from ocr import OCR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing OCR.')
    parser.add_argument('--image', help='Path to image file.')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # Open the image file
    if not os.path.isfile(args.image):
        logging.error("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)

    hasFrame, frame = cap.read()

    if hasFrame:
        images = {}
        images['frame'] = frame
        
        characteres, img, mask = extract_chars(frame) 
        images['img'] = img.copy()
        img = cv_skeletonize(img)
        img = np.uint8(img / 255)
        images['skel'] = img.copy()

        ocr = OCR(model_filename="../config/attention_ocr_model.pth", use_cuda=False, threshold=0.7)
        pred = ocr.predict(img)
        logging.info(f'Prediction: {pred}')

        plot_images(images, 1, 3, cmap='gray')
        
    else:
        logging.debug("Frame not found!")