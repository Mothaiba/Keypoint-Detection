# detect lines by Probabilistic Hough Transform

import cv2
import numpy as np

image = cv2.imread('screenshot.png', 0)
edge_image = cv2.Canny(image, 40, 70)

cv2.imshow('edges', edge_image)
# cv2.waitKey()

lines = cv2.HoughLinesP(edge_image, 1, np.pi / 180., 30, maxLineGap=5, minLineLength=10)
print 'There are {} lines'.format(len(lines))

image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for line in lines:
    (x1, y1, x2, y2) = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0))

cv2.imshow('lines', image)
cv2.waitKey()