from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
import base64

import logging

from image_preprocessing import extract_chars
from yolo import Yolo
from ocr import OCR

import importlib
tesseract_spec = importlib.util.find_spec("pytesseract")
tesseract_found = tesseract_spec is not None
if tesseract_found:
    import pytesseract

app = Flask(__name__)

DEBUG = True

@app.route('/')
def index():
    return "Live and Running!"

@app.route('/', methods=['POST'])
def run_lpr():
    logging.info(f'Debug mode {app.debug}')

    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        file.close()

        if(img_bytes is not None):
            nparr = np.fromstring(img_bytes, np.uint8)
            inputImage = cv.imdecode(nparr, cv.IMREAD_COLOR)

            # TODO: state management: avoid loading net for every request
            yolo = Yolo(img_width=1056, img_height=576, 
                        debug=DEBUG, confidence_threshold=0.6, non_max_supress_theshold=0.4,
                        classes_filename='../config/classes.names',
                        model_architecture_filename="../config/yolov3_license_plates.cfg", 
                        model_weights_filename="../config/yolov3_license_plates_last.weights",
                        output_directory='../debug/')                   

            roi_imgs = yolo.detect(inputImage)

            ocr = OCR(model_filename="../config/emnist_net_custom.pt", num_classes=36, use_cuda=False, debug=DEBUG)

            index = 0
            for roi_img in roi_imgs:
                logging.info(f'\n\nProcessing ROI {index}')
                box = [yolo.bounding_boxes[index][0], yolo.bounding_boxes[index][1], yolo.bounding_boxes[index][2], yolo.bounding_boxes[index][3]]
                predict(yolo.img, roi_img, box, str(index), (0,255,0), ocr)
                
                if(DEBUG):
                    cv.imwrite("../debug/roi_"+str(index)+".jpg", roi_img.astype(np.uint8))

                index += 1

            # API response: the highest confidence one
            logging.info(f'\n\n---Processing the Highest Confidence ROI---\n')
            bounding_box = None
            emnist_net_preds = None
            tesseract_preds = None
            if(yolo.highest_object_confidence > 0 and yolo.roi_img is not None):
                bounding_box = {
                    'x': yolo.box_x,
                    'y': yolo.box_y,
                    'w': yolo.box_w,
                    'h': yolo.box_h
                }                                
                _, emnist_net_preds, tesseract_preds = predict(yolo.img, yolo.roi_img, [yolo.box_x, yolo.box_y, yolo.box_w, yolo.box_h], "", (255,255,0), ocr)                                
                if(DEBUG):                    
                    cv.imwrite("../debug/result.jpg", yolo.img.astype(np.uint8))
    
            data = {
                'bounding_box': bounding_box,
                'confidence': yolo.highest_object_confidence,
                'classId': str(yolo.classId_highest_object),
                'emnist_net_preds': emnist_net_preds,
                'tesseract_preds': tesseract_preds
            }
            response = jsonify(data)

    response.status_code = 200
    return response

def predict(input_image, roi_img, bounding_box, prefix_label, background_color, emnist_net):
    characteres, img, mask = extract_chars(roi_img)
    
    emnist_net_preds = emnist_net.predict(characteres)

    tesseract_preds = None
    if tesseract_found:
        tesseract_preds = pytesseract.image_to_string(img, lang='eng', config='--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    logging.debug(f'\nTesseract output: {tesseract_preds}\nEMNISTNet output: {emnist_net_preds}')

    text = tesseract_preds if tesseract_preds is not None else emnist_net_preds
    labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = bounding_box[0]
    y = bounding_box[1] + bounding_box[3]
    w = bounding_box[0] + round(1.1*labelSize[0])
    h = (bounding_box[1] + bounding_box[3]) + 25
    
    cv.rectangle(input_image, (x, y), (w, h), background_color, cv.FILLED)                    
    cv.putText(input_image, text, (x+5, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if(DEBUG):
        cv.imwrite("../debug/roi_masked_"+prefix_label+".jpg", img.astype(np.uint8))
        cv.imwrite("../debug/roi_mask_"+prefix_label+".jpg", mask.astype(np.uint8))

    return characteres, emnist_net_preds, tesseract_preds

if __name__ == '__main__':

    if(DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    app.run(debug=DEBUG)
