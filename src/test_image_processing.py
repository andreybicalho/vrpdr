import cv2 as cv
import numpy as np
import argparse
import sys
import os.path
import logging
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, black_tophat, white_tophat
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
    # black tophat
    images['black_tophat 3x3'] = black_tophat(img_gray, square(3))
    images['black_tophat 5x5'] = black_tophat(img_gray, square(5))
    images['black_tophat 7x7'] = black_tophat(img_gray, square(7))
    images['black_tophat 9x9'] = black_tophat(img_gray, square(9))
    images['black_tophat 11x11'] = black_tophat(img_gray, square(11))
    # white tophat
    images['white_tophat 3x3'] = white_tophat(img_gray, square(3))
    images['white_tophat 5x5'] = white_tophat(img_gray, square(5))
    images['white_tophat 7x7'] = white_tophat(img_gray, square(7))
    images['white_tophat 9x9'] = white_tophat(img_gray, square(9))
    images['white_tophat 11x11'] = white_tophat(img_gray, square(11))
    plot_images(images, 9, 5, cmap="gray")

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

def test_skeleton_marker_based_watershed_segmentation(image):
    images = {}
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images['gray'] = img_gray.copy()
    ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    images['threshold inv'] = thresh.copy()
    output_img = thresh.copy()

    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    pre_marker_img = cv.morphologyEx(img_gray, cv.MORPH_BLACKHAT, structuringElement)
    images['black hat'] = pre_marker_img.copy()

    ret, pre_marker_img = cv.threshold(pre_marker_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    images['pre marker threshold'] = pre_marker_img.copy()

    skeleton = cv_skeletonize(pre_marker_img)
    images['skeleton'] = skeleton.copy()
    ret, markers = cv.connectedComponents(skeleton)
    images['markers'] = markers.copy()
    watershed_result = cv.watershed(image, markers)
    images['watershed_result'] = watershed_result.copy()

    watershed_result[watershed_result == -1] = 255
    watershed_result[watershed_result != 255] = 0
    watershed_result = np.uint8(watershed_result)
    images['final watershed_result'] = watershed_result.copy()

    _, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    images['mask'] = mask.copy()

    thresh[mask == 0] = 0
    images['thresh masked'] = thresh.copy()

    # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
    char_contours, refined_mask = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    images['refined_mask'] = refined_mask.copy()

    output_img[refined_mask == 0] = 0
    images['thresh refine masked'] = output_img.copy()
    output_img = util.invert(output_img)
    images['final result'] = output_img.copy()

    plot_images(images, 4, 4, cmap='gray')

def test_intersection_lines_marker_based_watershed_segmentation(image):
    images = {}
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    images['gray'] = img_gray.copy()
    ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    images['threshold inv'] = thresh.copy()
    output_img = thresh.copy()

    structuringElement = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    pre_marker_img = cv.morphologyEx(img_gray, cv.MORPH_BLACKHAT, structuringElement)
    images['black hat'] = pre_marker_img.copy()

    ret, pre_marker_img = cv.threshold(pre_marker_img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    images['pre marker threshold'] = pre_marker_img.copy()

    intersection_line_img = np.zeros(pre_marker_img.shape, np.uint8)
    height, width = pre_marker_img.shape

    cv.line(intersection_line_img, pt1=(0, int(height/2)), pt2=(width, int(height/2)), color=(255), thickness=5)
    cv.line(intersection_line_img, pt1=(0, int(height/2+height/4)), pt2=(width, int(height/2+height/4)), color=(255), thickness=5)
    images['lines'] = intersection_line_img.copy()
    intersection_img = cv.bitwise_and(intersection_line_img, pre_marker_img)
    images['lines and pre marker intersection'] = intersection_img.copy()

    ret, markers = cv.connectedComponents(intersection_img)
    images['markers'] = markers.copy()
    watershed_result = cv.watershed(image, markers)
    images['watershed_result'] = watershed_result.copy()

    watershed_result[watershed_result == -1] = 255
    watershed_result[watershed_result != 255] = 0
    watershed_result = np.uint8(watershed_result)
    images['final watershed_result'] = watershed_result.copy()    

    _, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    images['mask'] = mask.copy()

    thresh[mask == 0] = 0
    images['thresh masked'] = thresh.copy()

    # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
    char_contours, refined_mask = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    images['refined_mask'] = refined_mask.copy()

    output_img[refined_mask == 0] = 0
    images['thresh refine masked'] = output_img.copy()
    output_img = util.invert(output_img)
    images['final result'] = output_img.copy()

    plot_images(images, 4, 4, cmap='gray')

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
        #test_threshold_methods(image)
        #test_histogram_equalization_methods(image)
        #test_morphological_methods(frame)
        test_skeleton_marker_based_watershed_segmentation(frame)
        test_intersection_lines_marker_based_watershed_segmentation(frame)


        # comparing both methods
        images = {}
        image = frame
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images['skel gray'] = img_gray.copy()
        ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        images['skel threshold'] = thresh.copy()
        output_img = thresh.copy()

        # skeleton method
        watershed_result = skeleton_marker_based_watershed_segmentation(image)
        images['skel watershed'] = watershed_result.copy()        

        char_contours, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        thresh[mask == 0] = 0
        images['skel mask 1'] = mask.copy() 
        images['skel threshold masked 1'] = thresh.copy()

        # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
        char_contours, mask2 = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        images['skel mask 2'] = mask2.copy()
        output_img[mask2 == 0] = 0
        images['skel threshold masked 2'] = output_img.copy()

        output_img = util.invert(output_img)
        images['skel output'] = output_img.copy()

        # intersection lines method
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        images['intersec gray'] = img_gray.copy()
        ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
        images['intersec threshold'] = thresh.copy()
        output_img = thresh.copy()

        watershed_result = intersection_lines_marker_based_watershed_segmentation(image)
        images['intersec watershed'] = watershed_result.copy()

        char_contours, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        thresh[mask == 0] = 0
        images['intersec mask 1'] = mask.copy() 
        images['intersec threshold masked 1'] = thresh.copy()

        # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
        char_contours, mask2 = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
        images['intersec mask 2'] = mask2.copy()
        output_img[mask2 == 0] = 0
        images['intersec threshold masked 2'] = output_img.copy()

        output_img = util.invert(output_img)
        images['intersec output'] = output_img.copy()

        plot_images(images, 2, 8, cmap='gray')
        

    else:
        logging.debug("Frame not found!")