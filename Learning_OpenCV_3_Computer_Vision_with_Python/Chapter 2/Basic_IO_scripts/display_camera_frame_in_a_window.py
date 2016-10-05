# this program display the video captured from webcam
# press 'e' to exit the video
# click at a point in the screen will draw a small circle around it (not last-lasting)

import cv2
import numpy as np

def onMouseDrawCircle(event, x, y, flag, params):

    if event == cv2.EVENT_LBUTTONUP:
        global frame_idx, draw_frame_idx, pos_x, pos_y
        draw_frame_idx = frame_idx
        pos_x = x
        pos_y = y

cap = cv2.VideoCapture(0)
cv2.namedWindow('video')

cv2.setMouseCallback('video', on_mouse=onMouseDrawCircle)
frames_to_draw = 50
frame_idx = 0
draw_frame_idx = - frames_to_draw

while(True):
    status, image = cap.read()
    image = np.fliplr(image)
    frame_idx += 1

    if frame_idx - draw_frame_idx < frames_to_draw:
        image = cv2.circle(image, (pos_x, pos_y), 10, (0, 255, 0))
    cv2.imshow('video', image)

    key = cv2.waitKey(20)
    if key == ord('e'):
        break
