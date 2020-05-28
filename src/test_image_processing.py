import cv2 as cv
import numpy as np
import argparse
import sys
import os.path
import logging
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, black_tophat
from skimage.morphology import reconstruction
from skimage.morphology import disk, square, rectangle
from skimage.filters import threshold_li, threshold_mean, threshold_multiotsu, threshold_niblack, threshold_yen, threshold_otsu, threshold_local, threshold_sauvola, threshold_niblack, rank
from skimage import exposure
from skimage import util
from skimage import data
from skimage import feature
from imutils import contours
from image_processing import opening_by_reconstruction, closing_by_reconstruction, display_images, plot_images, cv_skeletonize, extract_contours, skeleton_marker_based_watershed_segmentation, intersection_lines_marker_based_watershed_segmentation


def test_threshold_methods(img):
    images = {}
    img_gray = rgb2gray(img)
    images['grayscale'] = img_gray
    images['threshold_local_5'] = threshold_local(img_gray, block_size=5)
    images['threshold_local_11'] = threshold_local(img_gray, block_size=11)    
    th = threshold_multiotsu(img_gray)
    images['threshold_multiotsu'] = np.digitize(img_gray, bins=th)
    th = threshold_otsu(img_gray)
    images['threshold_otsu'] = img_gray >= th
    th = threshold_li(img_gray)
    images['threshold_li'] = img_gray >= th
    th = threshold_yen(img_gray)
    images['threshold_yen'] = img_gray >= th
    th = threshold_mean(img_gray)
    images['threshold_mean'] = img_gray > th
    th = threshold_niblack(img_gray, window_size=25, k=0.8)
    images['thresh_niblack'] = img_gray > th
    th = threshold_sauvola(img_gray, window_size=25)
    images['threshold_sauvola'] = img_gray > th

    plot_images(images, 4, 4, cmap='gray')

def test_morphological_methods(image):    
    images = {}
    img_gray = rgb2gray(image)
    images['grayscale'] = img_gray
    images['grayscale 1'] = img_gray
    images['grayscale 2'] = img_gray
    images['grayscale 3'] = img_gray
    images['grayscale 4'] = img_gray
    # openings
    images['opening 3x3'] = opening(img_gray, square(3))
    images['opening 5x5'] = opening(img_gray, square(5))
    images['opening 7x7'] = opening(img_gray, square(7))
    images['opening 9x9'] = opening(img_gray, square(9))
    images['opening 11x11'] = opening(img_gray, square(11))
    # closings
    images['closing 3x3'] = closing(img_gray, square(3))
    images['closing 5x5'] = closing(img_gray, square(5))
    images['closing 7x7'] = closing(img_gray, square(7))
    images['closing 9x9'] = closing(img_gray, square(9))
    images['closing 11x11'] = closing(img_gray, square(11))
    # openings by reconstruction
    images['opening_by_reconstruction 3x3'] = opening_by_reconstruction(img_gray, square(3))
    images['opening_by_reconstruction 5x5'] = opening_by_reconstruction(img_gray, square(5))
    images['opening_by_reconstruction 7x7'] = opening_by_reconstruction(img_gray, square(7))
    images['opening_by_reconstruction 9x9'] = opening_by_reconstruction(img_gray, square(9))
    images['opening_by_reconstruction 11x11'] = opening_by_reconstruction(img_gray, square(11))
    # closings by reconstruction
    images['closing_by_reconstruction 3x3'] = closing_by_reconstruction(img_gray, square(3))
    images['closing_by_reconstruction 5x5'] = closing_by_reconstruction(img_gray, square(5))
    images['closing_by_reconstruction 7x7'] = closing_by_reconstruction(img_gray, square(7))
    images['closing_by_reconstruction 9x9'] = closing_by_reconstruction(img_gray, square(9))
    images['closing_by_reconstruction 11x11'] = closing_by_reconstruction(img_gray, square(11))
    # erosion
    images['erosion 3x3'] = erosion(img_gray, square(3))
    images['erosion 5x5'] = erosion(img_gray, square(5))
    images['erosion 7x7'] = erosion(img_gray, square(7))
    images['erosion 9x9'] = erosion(img_gray, square(9))
    images['erosion 11x11'] = erosion(img_gray, square(11))
    # dilation
    images['dilation 3x3'] = dilation(img_gray, square(3))
    images['dilation 5x5'] = dilation(img_gray, square(5))
    images['dilation 7x7'] = dilation(img_gray, square(7))
    images['dilation 9x9'] = dilation(img_gray, square(9))
    images['dilation 11x11'] = dilation(img_gray, square(11))
    plot_images(images, 7, 5, cmap="gray")

def test_histogram_equalization_methods(image):
    images = {}
    
    #img_gray = rgb2gray(image)
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images['grayscale'] = img_gray

    global_eq = exposure.equalize_hist(img_gray)
    images['global'] = global_eq

    selem = disk(30)
    local_eq = rank.equalize(img_gray, selem=selem)
    images['local'] = local_eq

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
    images['adaptative'] = img_adapteq

    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(img_gray)
    images['clahe'] = clahe_img

    plot_images(images, 3, 3, cmap='gray')

def test_pipeline(image):
    logging.debug(f'frame image: {image.shape} ---> {image.dtype}')
    #test_threshold_methods(image)
    #test_morphological_methods(image)
    #test_histogram_equalization_methods(image)
    images = {}
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images['gray'] = img.copy()

    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    images['threshold'] = thresh.copy()
    output_img = thresh.copy()

    #element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    #open_img = cv.morphologyEx(thresh, cv.MORPH_OPEN, element)
    #images['opened'] = open_img.copy()

    intersection_line_img = np.zeros(img.shape, np.uint8)
    height, width = img.shape
    logging.debug(f'{width} x {height}')
         
    cv.line(intersection_line_img, pt1=(0, int(height/2)), pt2=(width, int(height/2)), color=(255), thickness=3)
    cv.line(intersection_line_img, pt1=(0, int(height/2+height/4)), pt2=(width, int(height/2+height/4)), color=(255), thickness=3)
    logging.debug(f'{intersection_line_img.shape}')
    images['intersection line'] = intersection_line_img.copy()

    intersection_img = cv.bitwise_and(intersection_line_img, thresh)
    #intersection_img = np.uint8(intersection_img)
    logging.debug(f'intersection image: {intersection_img.shape} ---> {intersection_img.dtype}')
    images['intersection'] = intersection_img.copy()

    ret, markers = cv.connectedComponents(intersection_img)
    images['markers'] = markers.copy()

    watershed_result = cv.watershed(image, markers)
    images['watershed result'] = watershed_result.copy()

    watershed_result[watershed_result == -1] = 255
    watershed_result[watershed_result != 255] = 0
    watershed_result = np.uint8(watershed_result)
    images['watershed preprocessed'] = watershed_result.copy()

    chars, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    #display_images(chars, 5, 5)
    images['mask'] = mask.copy()
        
    thresh[mask == 0] = 0
    images['threshold masked'] = thresh.copy()

    chars, mask2 = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.3)
    #display_images(chars, 5, 5)
    images['mask 2'] = mask2.copy()
    output_img[mask2 == 0] = 0

    output_img = util.invert(output_img)
    images['output'] = output_img.copy()

    plot_images(images, 6, 6, cmap='gray')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing Image Processing Algorithms.')
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
        test_pipeline(frame)

        images = {}
        image = frame
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images['gray'] = img_gray.copy()
        ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        images['threshold'] = thresh.copy()
        output_img = thresh.copy()

        watershed_result = skeleton_marker_based_watershed_segmentation(image, thresh)
        images['watershed skeleton'] = watershed_result.copy()
    
        watershed_result = intersection_lines_marker_based_watershed_segmentation(image, thresh)
        images['watershed intersection line'] = watershed_result.copy()

        char_contours, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        thresh[mask == 0] = 0
        images['mask 1'] = mask.copy() 
        images['threshold masked 1'] = thresh.copy()

        # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
        char_contours, mask2 = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        images['mask 2'] = mask2.copy()
        output_img[mask2 == 0] = 0
        images['threshold masked 2'] = output_img.copy()

        output_img = util.invert(output_img)
        images['output'] = output_img.copy()

        plot_images(images, 6, 6, cmap='gray')
        

    else:
        logging.debug("Frame not found!")