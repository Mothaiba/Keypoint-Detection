import cv2
import numpy as np


# img = np.zeros((200, 200), dtype=np.uint8)
# img = cv2.imread('screenshot.png', 0)
#
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
# cv2.imshow("contours", color)
# cv2.waitKey()
# cv2.destroyAllWindows()


import cv2
import numpy as np, sys

A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
    # cv2.imshow('G', G)
    # cv2.waitKey()

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in xrange(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in xrange(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)
    # print L
    # cv2.imshow('L', L)
    # cv2.waitKey()

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in xrange(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)
    # cv2.imshow('L', L)
    # cv2.waitKey()

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
    LS.append(ls)
    # cv2.imshow('ls', ls)
    # cv2.waitKey()

# now reconstruct
ls_ = LS[0]
for i in xrange(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])
    cv2.imshow('ls_', ls_)
    cv2.waitKey()

# image with direct connecting each half
real = np.hstack((A[:, :cols / 2], B[:, cols / 2:]))

cv2.imshow('Pyramid_blending2.jpg', ls_)
cv2.imshow('Direct_blending.jpg', real)
cv2.waitKey(0)