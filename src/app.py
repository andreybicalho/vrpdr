from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
import base64

import logging

from image_processing import extract_chars
from yolo import Yolo
from ocr import OCR

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

            ocr = OCR(model_filename="../config/attention_ocr_model.pth", use_cuda=False, threshold=0.7)

            index = 0
            api_output = []
            for roi_img in roi_imgs:
                logging.info(f'\n\nProcessing ROI {index}')
                box = [yolo.bounding_boxes[index][0], yolo.bounding_boxes[index][1], yolo.bounding_boxes[index][2], yolo.bounding_boxes[index][3]]
                score = yolo.confidences[index]
                pred = predict(yolo.img, roi_img, box, str(index), (0,255,0), ocr)

                output = {'bounding_box' : box, 'confidence' : score, 'ocr_pred' : pred}
                api_output.append(output)
                
                index += 1
                            
            if(DEBUG):                 
                cv.imwrite("../debug/result.jpg", yolo.img.astype(np.uint8))
            
            success, output_image = cv.imencode('.jpg', yolo.img)
            api_response = {
                'output_image' : base64.b64encode(output_image).decode('utf-8'),
                'detections' : api_output
            }
            response = jsonify(api_response)

    response.status_code = 200
    return response

def predict(input_image, roi_img, bounding_box, prefix_label, background_color, ocr):
    characteres, masked_img, mask = extract_chars(roi_img)
    
    pred = ocr.predict(masked_img)
    logging.debug(f'\nOCR output: {pred}')

    labelSize, baseLine = cv.getTextSize(pred, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = bounding_box[0]
    y = bounding_box[1] + bounding_box[3]
    w = bounding_box[0] + round(1.1*labelSize[0])
    h = (bounding_box[1] + bounding_box[3]) + 25
    
    cv.rectangle(input_image, (x, y), (w, h), background_color, cv.FILLED)                    
    cv.putText(input_image, pred, (x+5, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    if(DEBUG):
        cv.imwrite("../debug/roi_"+prefix_label+".jpg", roi_img.astype(np.uint8))
        cv.imwrite("../debug/roi_masked_"+prefix_label+".jpg", masked_img.astype(np.uint8))
        cv.imwrite("../debug/roi_mask_"+prefix_label+".jpg", mask.astype(np.uint8))

    return pred

if __name__ == '__main__':

    if(DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    app.run(debug=DEBUG)
