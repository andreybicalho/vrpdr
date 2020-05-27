import cv2 as cv
import numpy as np
import logging
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.morphology import erosion, dilation, opening, closing, black_tophat
from skimage.morphology import reconstruction
from skimage.morphology import disk, square, rectangle
from skimage.filters import threshold_li, threshold_mean, threshold_multiotsu, threshold_niblack, threshold_yen, threshold_otsu, threshold_local, rank
from skimage import exposure
from skimage import util
from skimage import data
from imutils import contours

def equalize_histogram(image):
    images = {}
    img_gray = image
    
    if(len(img_gray.shape) > 2):
        img_gray = rgb2gray(image)
        images['grayscale'] = img_gray

    global_eq = exposure.equalize_hist(img_gray)
    images['global'] = global_eq

    local_eq = rank.equalize(img_gray, selem=disk(30))
    images['local'] = local_eq    
    
    plot_images(images, 2, 3, cmap='gray')

    return local_eq

def opening_by_reconstruction(image, se):    
    eroded = erosion(image, se)
    reconstructed = reconstruction(eroded, image)
    return reconstructed

def closing_by_reconstruction(image, se, iterations=1):
    obr = opening_by_reconstruction(image, se)

    obr_inverted = util.invert(obr)
    obr_inverted_eroded = erosion(obr_inverted, se)
    obr_inverted_eroded_rec = reconstruction(obr_inverted_eroded, obr_inverted)
    obr_inverted_eroded_rec_inverted = util.invert(obr_inverted_eroded_rec)
    return obr_inverted_eroded_rec_inverted

def square_resize(img):
    """
    This function resize non square image to square one (height == width)
    :param img: input image as numpy array
    :return: numpy array
    """
    # image after making height equal to width
    squared_image = img
    # Get image height and width
    h = img.shape[0]
    w = img.shape[1]

    # In case height superior than width
    if h > w:
        diff = h-w
        if diff % 2 == 0:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = x1
        else:
            x1 = np.zeros(shape=(h, diff//2))
            x2 = np.zeros(shape=(h, (diff//2)+1))
        squared_image = np.concatenate((x1, img, x2), axis=1)

    # In case height inferior than width
    if h < w:
        diff = w-h
        if diff % 2 == 0:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = x1
        else:
            x1 = np.zeros(shape=(diff//2, w))
            x2 = np.zeros(shape=((diff//2)+1, w))
        squared_image = np.concatenate((x1, img, x2), axis=0)

    return squared_image

def plot_images(data, rows, cols, cmap='gray'):
    if(len(data) > 0):
        i = 0
        for title, image in data.items():
            #logging.debug(title)    
            plt.subplot(rows,cols,i+1),plt.imshow(image,cmap)
            plt.title(title)
            plt.xticks([]),plt.yticks([])
            i += 1
        plt.show()

def display_images(img_list, row, col):
    if(len(img_list) > 0):
        images = {}
        n = 0
        for img in img_list:
            n += 1
            images[str(n)] = img
        plot_images(images, row, col, cmap='gray')

def draw_bounding_box(image, text_label, startPoint_x, startPoint_y, endPoint_x, endPoint_y, color=(0, 255, 0), thickness=2):
        # draw rectangle
        cv.rectangle(image, (startPoint_x, startPoint_y), (endPoint_x, endPoint_y), color, thickness)

        # draw the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(text_label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(startPoint_y, labelSize[1])
        cv.rectangle(image, (startPoint_x, top - round(1.5*labelSize[1])), (startPoint_x + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        cv.putText(image, text_label, (startPoint_x, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

def cv_skeletonize(img):
    """
    Steps:
        1 - Starting off with an empty skeleton.
        2 - Computing the opening of the original image. Let’s call this open.
        3 - Substracting open from the original image. Let’s call this temp.
        4 - Eroding the original image and refining the skeleton by computing the union of the current skeleton and temp.
        5 - Repeat Steps 2–4 till the original image is completely eroded.
    """
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    # Step 1: Create an empty skeleton
    skel = np.zeros(img.shape, np.uint8)
    while True:
        #Step 2: Open the image
        open = cv.morphologyEx(img, cv.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv.erode(img, element)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv.countNonZero(img)==0:
            break

    return skel

def extract_contours(image, min_contours_area_ratio=0.02, max_contours_area_ratio=0.2):
    mask = np.zeros(image.shape, dtype=np.uint8)
    cnts = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    logging.debug(f'Found {len(cnts)} contours!')
    roi_index = 0
    total_area = image.shape[0] * image.shape[1]
    logging.debug(f'Total area ({image.shape[0]}, {image.shape[1]}): {total_area}')
    contours_used_for_masking = 0
    rois = []
    for c in cnts:        
        x,y,w,h = cv.boundingRect(c)
        roi_area = w * h
        roi_area_ratio = roi_area / total_area
        logging.debug(f'ROI {roi_index} area: {roi_area} - ratio: {roi_area_ratio}')
        
        if roi_area_ratio >= min_contours_area_ratio and roi_area_ratio <= max_contours_area_ratio:
            contours_used_for_masking += 1
            roi = image[y:y+h, x:x+w].copy()
            
            mask[y:y+h, x:x+w] = 255

            aux_roi = np.array(roi)
            aux_roi = cv.resize(aux_roi,(28,28), interpolation = cv.INTER_AREA)            
            aux_roi[aux_roi != 0] = 255
            rois.append(aux_roi)

        roi_index += 1

    logging.debug(f'Contours used for masking: {contours_used_for_masking}')

    return rois, mask

def marker_based_watershed_segmentation(image, pre_marker_img):    
    skeleton = cv_skeletonize(pre_marker_img)
    ret, markers = cv.connectedComponents(skeleton)
    watershed_result = cv.watershed(image, markers)
    
    return watershed_result
    
def extract_chars(image):
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    se = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    eroded = cv.erode(thresh, se, iterations=2)
    dilated = cv.dilate(eroded, se, iterations=2)

    watershed_result = marker_based_watershed_segmentation(image, dilated)

    watershed_result[watershed_result == -1] = 255
    watershed_result[watershed_result != 255] = 0
    watershed_result = np.uint8(watershed_result)

    _, mask = extract_contours(image=watershed_result, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)
    thresh[mask == 0] = 0    

    # we can run extract_contours again but this time on the threshold masked to get the char contours more accurate
    char_contours, _ = extract_contours(image=thresh, min_contours_area_ratio=0.01, max_contours_area_ratio=0.2)

    # now make the image properly for tesseract (white background)
    thresh = util.invert(thresh)

    return char_contours, thresh, mask