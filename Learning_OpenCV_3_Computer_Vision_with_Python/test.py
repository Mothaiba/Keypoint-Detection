import cv2
import numpy as np

import imutils

from os.path import join, isfile
from os import listdir

_test_path = './Chapter 7/Image_classification/TestImages/'
_test_imgs = [join(_test_path, f) for f in listdir(_test_path) if isfile(join(_test_path, f))]

for idx, _test_img in enumerate(_test_imgs[:1]):
    category = 'noncar'
    print category
    test_img = imutils.resize(cv2.imread(_test_img, 1), height=400)
    cv2.putText(test_img, category, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), cv2.LINE_4)

    cv2.imshow(str(idx), test_img)

cv2.waitKey()