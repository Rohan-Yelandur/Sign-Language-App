import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

video_capture = cv2.VideoCapture(0)
hand_detector = HandDetector(maxHands=1)
img_size = 300
offset = 20
counter = 0
folder = 'Data/A'

while True:
    success, img = video_capture.read()
    img_raw = img.copy()
    hands, img_drawn = hand_detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox'] # Info for bounding box of hand
        img_formatted = np.ones((img_size, img_size, 3), np.uint8) * 255
        
        try:
            img_crop = img_drawn[y - offset : y + h + offset, x - offset : x + w + offset]

            aspect_ratio  = h / w
            if aspect_ratio > 1:
                k = img_size / h
                w_calculated = math.ceil(k * w)
                img_resize = cv2.resize(img_crop, (w_calculated, img_size))
                w_gap = math.ceil((img_size - w_calculated) / 2)
                img_formatted[:, w_gap : w_calculated + w_gap] = img_resize
            else:
                k = img_size / w
                h_calculated = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (img_size, h_calculated))
                h_gap = math.ceil((img_size - h_calculated) / 2)
                img_formatted[h_gap : h_calculated + h_gap, :] = img_resize

            cv2.imshow("Image Cropped", img_crop)
            cv2.imshow("Image Formatted", img_formatted)
            if cv2.getWindowProperty('Image Raw', cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception as e:
            print(f"Hand out of bounds")

    cv2.imshow("Image Raw", img_raw)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        counter += 1
        cv2.imwrite(f"{folder}/Image_{counter}-{time.time()}.png", img_formatted)
    if cv2.getWindowProperty('Image Raw', cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()