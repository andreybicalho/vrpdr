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
from image_preprocessing import opening_by_reconstruction, closing_by_reconstruction, display_images, plot_images, extract_chars


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
    
    img_gray = rgb2gray(image)
    images['grayscale'] = img_gray

    global_eq = exposure.equalize_hist(img_gray)
    images['global'] = global_eq

    selem = disk(30)
    local_eq = rank.equalize(img_gray, selem=selem)
    images['local'] = local_eq    
    
    plot_images(images, 2, 3, cmap='gray')

def test_watershed(image):
    images = {}
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images['grayscale'] = img


    r = 3
    img = opening(img, disk(r))
    images['opening'+str(r)] = img

    w = 3
    h = 3
    img = closing_by_reconstruction(img, rectangle(w, h))
    images['closing_by_reconstruction_'+str(w)+'x'+str(h)] = img
    img = np.uint8(img)
    
    sigma=3
    img = feature.canny(img, sigma=sigma)
    images['canny_'+str(sigma)] = img

    img = np.uint8(img)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    images['threshold'] = thresh

    r = 5
    img = opening(img, disk(r))
    images['opening_'+str(r)] = img

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opened = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    images['opened'] = opened
    # sure background area
    sure_bg = cv.dilate(opened,kernel,iterations=3)
    images['sure_bg'] = sure_bg
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opened,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    images['dist_transform'] = dist_transform
    images['sure_fg'] = sure_fg
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    images['unknown'] = unknown
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    images['markers'] = markers
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    images['markers+1'] = markers
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    images['markers_unknown'] = markers
    markers = cv.watershed(image, markers)
    images['markers_watershed'] = markers
    image[markers == -1] = [255, 0, 0]
    images['image'] = image

    plot_images(images, 6, 3, cmap='gray')

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
        #test_threshold_methods(frame)
        #test_morphological_methods(frame)
        #test_histogram_equalization_methods(frame)
        #test_watershed(frame)
        extract_chars(frame, debug=True, prefix_label='test', min_countours_area_ratio=0.008, max_countours_area_ratio=0.1)
    else:
        logging.debug("Frame not found!")