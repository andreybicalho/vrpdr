import cv2 as cv
import numpy as np
import os.path
import logging

class Yolo:
    """ 
    Convenient way of using You Only Look Once (YOLO) object detector algorithm with OpenCV.
    """

    def __init__(self, img_width, img_height,
                classes_filename="config/classes.names", 
                model_architecture_filename="config/darknet-yolov3.cfg", model_weights_filename="config/darknet-yolov3-weights.weights", 
                output_directory="debug/", confidence_threshold=0.5, non_max_supress_theshold=0.4,
                output_image=True):

        self.img = None
        self.class_ids = []
        self.confidences = []
        self.bounding_boxes = []
        self.roi_img = None # region of interest (cropped image using the bounding box of the object of highest confidence)
        self.roi_imgs = []
        self.highest_object_confidence = -1.0
        self.classId_highest_object = None
        self.inference_time = -1.0
        self.output_directory = output_directory
        self.output_image = output_image
        
        self._classes = None
        self._inputImgWidth = img_width  # Width of network's input image
        self._inputImgHeight = img_height  # Height of network's input image
        self._confidence_threshold = confidence_threshold
        self._non_max_supress_theshold = non_max_supress_theshold # Non-maximum suppression threshold

        self._net = self.load_net(classes_filename, model_architecture_filename, model_weights_filename)


    def load_net(self, classes_filename, model_architecture_filename, model_weights_filename):
        logging.debug("Loading YOLO v3...")

        with open(classes_filename, 'rt') as f:
            self._classes = f.read().rstrip('\n').split('\n')
            logging.debug(f'Classes: {self._classes}' )

        net = cv.dnn.readNetFromDarknet(model_architecture_filename, model_weights_filename)
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        return net

    
    def _getOutputLayersNames(self):
        """
        Get the names of the output layers.
        """
        
        # Get the names of all the layers in the network
        layersNames = self._net.getLayerNames()
        
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in self._net.getUnconnectedOutLayers()]

    
    def non_max_supression(self, input_image, net_outputs, draw_bounding_box=True):
        """ 
        Remove the bounding boxes with low confidence using non-maximum suppression. 
        """
        logging.debug("\nRunning Non Maximum Suppression to remove low confidence boxes...\n")

        frameHeight = input_image.shape[0]
        frameWidth = input_image.shape[1]

        class_ids = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        net_output_count = 0
        for out in net_outputs:
            net_output_count += 1
            logging.debug(f'Net output {net_output_count} of shape {out.shape}')
            detection_count = 0
            for detection in out:
                detection_count += 1
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self._confidence_threshold:
                    logging.debug(f"Confidence for detection {detection_count} of the net output {net_output_count}: {confidence}")
                    logging.debug(detection)
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    _x = int(center_x - width / 2)
                    _y = int(center_y - height / 2)
                    class_ids.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([_x, _y, width, height])

        logging.debug(f"Number of boxes found: {len(boxes)}")
        # Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self._confidence_threshold, self._non_max_supress_theshold) 
        
        for i in indices:
            i = i[0]
            box = boxes[i]
            conf = confidences[i]
            classId = class_ids[i]
            
            self.bounding_boxes.append(box)
            self.class_ids.append(classId)
            self.confidences.append(conf)

            box_x = box[0]
            box_y = box[1]
            box_w = box[2]
            box_h = box[3]
            
            if draw_bounding_box:
                self.draw_bounding_box(self.img, classId, conf, box_x, box_y, box_x + box_w, box_y + box_h)
            
            # keep a special one: the one with the highest confidence            
            if conf > self.highest_object_confidence:
                self.highest_object_confidence = conf
                box = boxes[i]
                self.box_x = box_x
                self.box_y = box_y
                self.box_w = box_w
                self.box_h = box_h
                self.classId_highest_object = classId


    def detect(self, input_image, draw_bounding_box=True):
        logging.debug("\nYOLO object detector is running...")

        self.img = input_image

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(input_image, 1/255, (self._inputImgWidth, self._inputImgHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self._net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        netOutputs = self._net.forward(self._getOutputLayersNames())

        # The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = self._net.getPerfProfile()
        self.inference_time = t * 1000.0 / cv.getTickFrequency()
        logging.debug(f"Inference time: {self.inference_time} ms")

        # Remove the bounding boxes with low confidence
        self.non_max_supression(self.img, netOutputs, draw_bounding_box)        

        # get roi for every bounding box
        if len(self.confidences) > 0:
            for box in self.bounding_boxes:
                box_x = box[0]
                box_y = box[1]
                box_w = box[2]
                box_h = box[3]
                self.roi_imgs.append(input_image[box_y:box_y+box_h, box_x:box_x+box_w].copy())                    

        # the highest confidence ROI
        #if(self.highest_object_confidence > 0):
        #    # ROI
        #    self.roi_img = input_image[self.box_y:self.box_y+self.box_h, self.box_x:self.box_x+self.box_w].copy()
        #    self.draw_bounding_box(self.img, self.classId_highest_object, self.highest_object_confidence, self.box_x, self.box_y, self.box_x + self.box_w, self.box_y + self.box_h, color=(255,255,0), thickness=2)            
        
        if self.output_image:
            cv.imwrite(self.output_directory+"YOLO.jpg", self.img.astype(np.uint8))          

        return self.roi_imgs


    def draw_bounding_box(self, image, class_id, confidence, start_point_x, start_point_y, end_point_x, end_point_y, color=(0, 255, 0), thickness=2):
        cv.rectangle(image, (start_point_x, start_point_y), (end_point_x, end_point_y), color, thickness)

        confidence *= 100
        label = '%.2f' % confidence + '%'        

        # Get the label for the class name and its confidence
        if self._classes:
            assert(class_id < len(self._classes))
            label = '%s: %s' % (self._classes[class_id], label)

        # draw the label at the top of the bounding box
        label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
        top = max(start_point_y, label_size[1])
        cv.rectangle(image, (start_point_x, top - round(1.1*label_size[1])), (start_point_x + round(1.1*label_size[0]), top + base_line), color, cv.FILLED)
        cv.putText(image, label, (start_point_x, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness)

    def clear(self):
        self.img = None
        self.class_ids = []
        self.confidences = []
        self.bounding_boxes = []
        self.roi_img = None
        self.roi_imgs = []
        self.highest_object_confidence = -1.0
        self.classId_highest_object = None
        self.inference_time = -1.0