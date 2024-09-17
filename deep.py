from os import listdir
from os.path import join, splitext
import logging

from timeit import default_timer
import random

import torch
import numpy as np
import cv2

from brickenator import DuploBrick
import re

CUDA_IS_AVALIABLE = False
if torch.cuda.is_available():
    CUDA_IS_AVALIABLE = True
else:
    logging.warning("CUDA is not avaliable, expect extremely low FPS!")
    torch.set_num_threads(4) # Prevent rampant threading on windows

class YoloBrickDetector():
    CLASSES = ['grey8', 'black16', 'blue8', 'cyan24', 'orange4', 'orange8', 'yellow4', 'yellow6', 'yellow8', 'lime4', 'lime8', 'green16']

    def __init__(self, modelPath):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=modelPath).autoshape()  # for PIL/cv2/np inputs and NMS

        if CUDA_IS_AVALIABLE:
            self.model = self.model.cuda()
        else:
            logging.warn("CUDA is not avaliable, expect extremely low FPS!")

        # Warmup model
        _ = self.model(np.zeros((480, 640, 3), dtype=np.uint8))

    def name(self):
        return "Approach B"

    def __call__(self, src, hook = None):
        time_start = default_timer()

        out = src.copy()

        debug = None
        if isinstance(hook, str):
            if hook == "output":
                hook = None
            else:
                debug = src.copy()

        # Inference
        results = self.model(out[:,:,::-1])  # includes NMS
        bricks = []

        # A large portion of the hook drawing code is referenced from the original yolov5 drawing code.
        random.seed(len(results.pred[0]))
        for *xyxy, conf, clazz in results.pred[0]:
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

            name = self.CLASSES[int(clazz)]
            if hook == "predictions":
                labelname = f'{name} {conf:.2%}'
                labelcolor = [random.randint(0, 127) for _ in range(3)]
                tl = 2
                cv2.rectangle(debug, c1, c2, color=labelcolor, thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(labelname, 0, fontScale=tl / 3, thickness=tf)[0]
                c3 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(debug, c1, c3, labelcolor, cv2.FILLED, cv2.LINE_AA)  # Text bg box
                cv2.putText(debug, labelname, (c1[0], c1[1] - 2), 0, tl / 3, color=(255,255,255), thickness=tf, lineType=cv2.LINE_AA)

            if conf < 0.8:
                continue

            x,y,w,h = cv2.boundingRect(np.array([c1, c2]))
            bbox = np.int0([(x,y), (x+w, y), (x+w, y+h), (x, y+h)])
            o = np.mean(bbox, axis=0).astype('uint32')

            properties = re.split('(\d+)', name)

            prediction = DuploBrick(points=bbox, center=tuple(o), color=properties[0].upper(), circleCount=int(properties[1]))

            bricks.append(prediction)

        time_end = default_timer()

        fps = 1/(time_end - time_start)

        cv2.putText(out, f'FPS: {fps:.1f}', (0,16), cv2.FONT_HERSHEY_PLAIN, 1, (225, 255, 255), 1, lineType=cv2.LINE_AA)

        return bricks, out, debug
            