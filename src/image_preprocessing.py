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

def closing_by_reconstruction(image, se):    
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

def extract_chars(image, debug=False, prefix_label=1, min_countours_area_ration=0.02, max_countours_area_ration=0.1):
    images = {}
    img_gray = image
    output_directory = "../debug/"+prefix_label+"_"
    
    if(len(image.shape) > 2):
        img_gray = rgb2gray(image)
        #images['img_gray'] = img_gray

    if(debug):
        img_out = img_gray.copy()
        img_out *= 255
        cv.imwrite(output_directory+"gray.jpg", img_out.astype(np.uint8))

    x = 3
    y = 3
    cbr = closing_by_reconstruction(img_gray, rectangle(x, y))
    #images['closing_by_reconstruction_'+str(x)+'x'+str(y)] = cbr.copy()
    if(debug):
        img_out = cbr.copy()
        img_out *= 255
        cv.imwrite(output_directory+"closing_by_reconstruction.jpg", img_out)

    th = threshold_li(cbr)
    thresh = cbr >= th
    thresh = np.uint8(thresh)
    
    #images['threshold'] = thresh.copy()
    if(debug):
        img_out = thresh.copy()
        img_out *= 255
        cv.imwrite(output_directory+"threshold.jpg", img_out)

    mask = np.ones(thresh.shape, dtype=np.uint8)
    mask *= 255
    #images['mask init'] = mask.copy()

    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
    logging.debug(f'Found {len(cnts)} contours!')
    ROI_number = 0
    chars = []
    total_area = thresh.shape[0] * thresh.shape[1]
    roi_area_threshold = total_area * max_countours_area_ration
    logging.debug(f'Total area ({thresh.shape[0]}, {thresh.shape[1]}): {total_area} - roi area threshold: {roi_area_threshold}')
    contours_used_for_masking = 0
    for c in cnts:
        # one single contour should have at maximum about max_countours_area_ratio (?? 10% ??) of the total image area and minimum of min_countours_area_ration (?? 2% ??)
        area = cv.contourArea(c)        
        roi_area_ration = area / total_area
        logging.info(f'ROI {ROI_number} area: {area} - ratio: {roi_area_ration}')
        if roi_area_ration >= min_countours_area_ration and roi_area_ration <= max_countours_area_ration:
            contours_used_for_masking += 1
            x,y,w,h = cv.boundingRect(c)
            #ROI = 255 - thresh[y:y+h, x:x+w]
            ROI = thresh[y:y+h, x:x+w].copy()
            #images['ROI_'+str(ROI_number)] = ROI.copy()
            
            mask[y:y+h, x:x+w] = 0
            #images['mask_added_ROI_'+str(ROI_number)] = mask.copy()

            ch = np.array(ROI)
            #images['ch_array_'+str(ROI_number)] = ch
            ch = cv.resize(ch,(28,28), interpolation = cv.INTER_AREA)            
            ch[ch != 0] = 255
            ch = util.invert(ch)
            #images['ch_'+str(ROI_number)] = ch
            chars.append(ch)

            if(debug):
                cv.imwrite(output_directory+"char_"+str(ROI_number)+".jpg", ch)

        ROI_number += 1
    
    logging.info(f'Contours used for masking: {contours_used_for_masking}')

    #images['mask'] = mask.copy()

    if(debug):
        img_out = mask.copy()
        cv.imwrite(output_directory+"mask.jpg", img_out)
    
    thresh *= 255
    thresh[mask != 0] = 255
    #images['threshold_masked'] = thresh.copy()

    if(debug):
        img_out = thresh.copy()
        cv.imwrite(output_directory+"threshold_masked.jpg", img_out)    

    plot_images(images, 7, 5, cmap='gray')

    return thresh, chars