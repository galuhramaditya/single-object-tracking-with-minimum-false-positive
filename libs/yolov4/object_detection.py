import numpy as np
import cv2

class Detector:
    def __init__(self, cfg, weight, confThreshold=0.5, nmsThreshold=0.4, only_class=[]):
        net = cv2.dnn.readNetFromDarknet(cfg, weight)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        
        self.only_class    = only_class
        self.confThreshold = confThreshold
        self.nmsThreshold  = nmsThreshold
        self.model         = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1/255, size=(416, 416), swapRB=True)
    
    def detect(self, frame):
        classIds, scores, boxes = self.model.detect(frame, confThreshold=self.confThreshold, nmsThreshold=self.nmsThreshold)
        if len(boxes) > 0:
            boxes = np.hstack([boxes[:, :2], boxes[:, :2]+boxes[:, 2:]])
        
        if len(self.only_class) > 0 and len(classIds) > 0: 
            classIds = np.argwhere(np.isin(classIds, self.only_class))[:, 0]
            scores   = scores[classIds]
            boxes    = boxes[classIds]
            return scores, boxes

        return scores, boxes