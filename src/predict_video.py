import numpy as np
import cv2 as cv
import numpy as np
import argparse
import sys
import os.path

from yolo import Yolo
from ocr import OCR

def draw_bounding_box(input_image, bounding_box, label, background_color, ocr):
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = bounding_box[0]
    y = bounding_box[1] + bounding_box[3]
    w = bounding_box[0] + round(1.1*labelSize[0])
    h = (bounding_box[1] + bounding_box[3]) + 25
    
    cv.rectangle(input_image, (x, y), (w, h), background_color, cv.FILLED)                    
    cv.putText(input_image, label, (x+5, y+20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python predict_video.py --input=path/to/the/video/file.mp4 --out=output.avi')
    parser.add_argument('--input', help='Input video file.')
    parser.add_argument('--out', help='Name of the output video file.')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input video file ", args.input, " not found!")
        sys.exit(1)

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(args.out, fourcc, 20.0, (1920,1080), True)

    yolo = Yolo(img_width=1056, img_height=576, 
                confidence_threshold=0.6, non_max_supress_theshold=0.4,
                classes_filename='../config/classes.names',
                model_architecture_filename="../config/yolov3_license_plates.cfg", 
                model_weights_filename="../config/yolov3_license_plates_last.weights",
                output_directory='../debug/',
                output_image=False)                   
    
    ocr = OCR(model_filename="../config/attention_ocr_model.pth", use_cuda=False, threshold=0.7)

    cap = cv.VideoCapture(args.input)

    while(cap.isOpened()):
        hasFrame, frame = cap.read()
        if hasFrame:
            roi_imgs = yolo.detect(frame)
            index = 0
            for roi_img in roi_imgs:
                box = [yolo.bounding_boxes[index][0], yolo.bounding_boxes[index][1], yolo.bounding_boxes[index][2], yolo.bounding_boxes[index][3]]
                pred = ocr.predict(roi_img)
                draw_bounding_box(input_image=frame, bounding_box=box, label=pred, background_color=(0,255,0), ocr=ocr)
                index += 1

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imshow('frame', frame)
            out.write(frame)
            yolo.clear()
        else:
            break

    # Release everything
    cv.destroyAllWindows()
    cap.release()
    out.release()
