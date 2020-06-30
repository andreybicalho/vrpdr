import numpy as np
import cv2 as cv
import numpy as np
import argparse
import sys
import os.path

from yolo import Yolo
from ocr import OCR
from app import draw_bounding_box

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: python predict_video.py --input=path/to/the/video/file.mp4 --out=output.avi')
    parser.add_argument('--input', help='Input video file.')
    parser.add_argument('--out', help='Name of the output video file.')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print("Input video file ", args.input, " not found!")
        sys.exit(1)

    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    out = cv.VideoWriter(args.out, fourcc, 30.0, (1920,1080), True)

    yolo = Yolo(img_width=1056, img_height=576, 
                confidence_threshold=0.85, non_max_supress_theshold=0.7,
                classes_filename='../config/classes.names',
                model_architecture_filename="../config/yolov3_license_plates.cfg", 
                model_weights_filename="../config/yolov3_license_plates_last.weights",
                output_directory='../debug/',
                output_image=False)                   
    
    ocr = OCR(model_filename="../config/attention_ocr_model.pth", use_cuda=False)

    cap = cv.VideoCapture(args.input)

    frame_count = 0
    while(cap.isOpened()):
        hasFrame, frame = cap.read()
        if hasFrame:
            frame_count += 1
            if frame_count % 2 == 0: # process every other frame to save time
                roi_imgs = yolo.detect(frame, draw_bounding_box=False)
                index = 0
                for roi_img in roi_imgs:
                    box = [yolo.bounding_boxes[index][0], yolo.bounding_boxes[index][1], yolo.bounding_boxes[index][2], yolo.bounding_boxes[index][3]]
                    score = yolo.confidences[index]
                    class_id = yolo.class_ids[index]
                    pred = ocr.predict(roi_img)
                    draw_bounding_box(input_image=frame, class_id=class_id, bounding_box=box, box_score=score, prediction=pred, background_color=(0,255,0))
                    index += 1

                out.write(frame)
                yolo.clear()

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            cv.imshow('frame', frame)
        else:
            break

    cv.destroyAllWindows()
    cap.release()
    out.release()
