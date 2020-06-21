from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
import base64

import logging

from yolo import Yolo
from ocr import OCR

app = Flask(__name__)

DEBUG = True

# TODO: state management and how to handle multiple request on this?
yolo = Yolo(img_width=1056, img_height=576, 
            confidence_threshold=0.6, non_max_supress_theshold=0.4,
            classes_filename='../config/classes.names',
            model_architecture_filename="../config/yolov3_license_plates.cfg", 
            model_weights_filename="../config/yolov3_license_plates_last.weights",
            output_directory='../debug/',
            output_image=True)

ocr = OCR(model_filename="../config/attention_ocr_model.pth", use_cuda=False, threshold=0.7)

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

            roi_imgs = yolo.detect(inputImage, draw_bounding_box=False)

            index = 0
            api_output = []
            for roi_img in roi_imgs:
                logging.info(f'Processing ROI {index}')
                box = [yolo.bounding_boxes[index][0], yolo.bounding_boxes[index][1], yolo.bounding_boxes[index][2], yolo.bounding_boxes[index][3]]
                score = yolo.confidences[index]
                class_id = yolo.class_ids[index]
                
                pred = ocr.predict(roi_img)

                draw_bounding_box(input_image=yolo.img, class_id=class_id, bounding_box=box, box_score=score, prediction=pred, background_color=(0,255,0))
                logging.info(f'OCR output: {pred}')
                    
                output = {'bounding_box' : box, 'bb_confidence' : score, 'ocr_pred' : pred}
                api_output.append(output)

                if(DEBUG):
                    cv.imwrite("../debug/roi_"+str(index)+".jpg", roi_img.astype(np.uint8))
                
                index += 1
                            
            if(DEBUG):                 
                cv.imwrite("../debug/result.jpg", yolo.img.astype(np.uint8))
            
            success, output_image = cv.imencode('.jpg', yolo.img)
            api_response = {
                'output_image' : base64.b64encode(output_image).decode('utf-8'),
                'detections' : api_output
            }
            response = jsonify(api_response)
    
            yolo.clear()

    response.status_code = 200
    return response

def draw_bounding_box(input_image, class_id, bounding_box, box_score, prediction, background_color):
    labelSize, baseLine = cv.getTextSize(prediction, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = bounding_box[0]
    y = bounding_box[1] + bounding_box[3]
    w = bounding_box[0] + round(1.1*labelSize[0])
    h = (bounding_box[1] + bounding_box[3]) + 25
    
    yolo.draw_bounding_box(input_image, class_id, box_score, bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3])

    cv.rectangle(input_image, (x, y), (w, h), background_color, cv.FILLED)                    
    cv.putText(input_image, prediction, (x+5, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

if __name__ == '__main__':

    if(DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    app.run(debug=DEBUG)
