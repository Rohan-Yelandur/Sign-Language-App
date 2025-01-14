# Suppress TensorFlow warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

class DataCollector:
    def __init__(self, folder, img_size, offset):
        self.video_capture = cv2.VideoCapture(0)
        self.img_size = img_size
        self.offset = offset
        self.counter = 0
        self.folder = folder
        self.hand_detector = HandDetector(maxHands=1)  # Create HandDetector instance once

    def collect_data(self):
        while True:
            success, img = self.video_capture.read()
            img_raw = img.copy()
            hands, img_drawn = self.hand_detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']  # Info for bounding box of hand
                img_formatted = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255

                try:
                    img_crop = img_drawn[y - self.offset : y + h + self.offset, x - self.offset : x + w + self.offset]

                    aspect_ratio = h / w
                    if aspect_ratio > 1:
                        k = self.img_size / h
                        w_calculated = math.ceil(k * w)
                        img_resize = cv2.resize(img_crop, (w_calculated, self.img_size))
                        w_gap = math.ceil((self.img_size - w_calculated) / 2)
                        img_formatted[:, w_gap : w_calculated + w_gap] = img_resize
                    else:
                        k = self.img_size / w
                        h_calculated = math.ceil(k * h)
                        img_resize = cv2.resize(img_crop, (self.img_size, h_calculated))
                        h_gap = math.ceil((self.img_size - h_calculated) / 2)
                        img_formatted[h_gap : h_calculated + h_gap, :] = img_resize

                    cv2.imshow('Image Cropped', img_crop)
                    cv2.imshow('Image Formatted', img_formatted)
                except Exception as e:
                    pass

            cv2.imshow('Image Raw', img_raw)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.counter += 1
                cv2.imwrite(f'{self.folder}/Image_{self.counter}-{time.time()}.png', img_formatted)
            if cv2.getWindowProperty('Image Raw', cv2.WND_PROP_VISIBLE) < 1:
                break

        self.video_capture.release()
        cv2.destroyAllWindows()