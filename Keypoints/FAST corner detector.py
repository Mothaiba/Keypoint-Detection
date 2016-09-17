import cv2
import numpy as np

# Has NOT implement non-max suppression
class FAST:

    def __init__(self, t=40):
        # n contiguous pixels of the 16 pixels around must be all
        # brighter or darker than the examining pixel
        self.n = 12
        # threshold t for brighter and darker
        self.t = t
        # relative positions of the 16 pixels around
        self.circle_around = [(-3, 0), (-3, 1), (-2, 2), (-1, 3), (0, 3), (1, 3),
                              (2, 2), (3, 1), (3, 0), (3, -1), (2, -2), (1, -3),
                              (0, -3), (-1, -3), (-2, -2), (-3, -1)]

    def detect_corners(self, img, nonMaxSuppression = False):

        # if input image has colours, convert it to grayscale
        if len(img.shape) > 2 or img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # set currently examining image
        self.img = img
        self.rows, self.cols = img.shape[:2]

        # define a matrix to indicate which pixels are candidate corners (before non-max suppression)
        self.corner_matrix = np.zeros(img.shape[:2], dtype=bool)

        # search for corners
        for i in range(3, self.rows - 3):
            for j in range(3, self.cols -3):
                if self.high_speed_test(i, j) and self.full_test(i, j):
                # if self.full_test(i, j):
                    self.corner_matrix[i, j] = True

        if nonMaxSuppression:
            self.doSuppression()

        x_corners, y_corners =  np.where(self.corner_matrix == True)
        corners = np.vstack((x_corners.T, y_corners.T)).T

        return corners

    def high_speed_test(self, x, y):
        self.base_inten_min = self.img[x, y] - self.t
        self.base_inten_max = self.img[x, y] + self.t
        self.status_around = np.zeros(4)

        # check the first and the 9-th pixel around
        if self.img[x - 3, y] <= self.base_inten_min:
            self.status_around[0] = -1
        elif self.img[x - 3, y] >= self.base_inten_max:
            self.status_around[0] = 1

        if self.img[x + 3, y] <= self.base_inten_min:
            self.status_around[1] = -1
        elif self.img[x + 3, y] >= self.base_inten_max:
            self.status_around[1] = 1

        # if both are not brighter or darker, return this is not a corner
        if self.status_around[0] == 0 and self.status_around[1] == 0:
            return False

        # check the 5-th and the 13-th pixel around
        if self.img[x, y + 3] <= self.base_inten_min:
            self.status_around[2] = -1
        elif self.img[x, y + 3] >= self.base_inten_max:
            self.status_around[2] = 1

        if self.img[x, y - 3] <= self.base_inten_min:
            self.status_around[3] = -1
        elif self.img[x, y - 3] >= self.base_inten_max:
            self.status_around[3] = 1

        # if at least 3 out of 4 are brighter or darker than than the
        # examining pixel, this maybe a corner
        if np.sum(self.status_around > 0) >= 3 or np.sum(self.status_around < 0) >= 3:
            return True
        # else, it is definitely not
        return False

    def full_test(self, x, y):
        self.base_inten_min = self.img[x, y] - self.t
        self.base_inten_max = self.img[x, y] + self.t
        self.status_around = np.zeros(16, dtype='int8')

        for i in range(16):
            if self.img[x + self.circle_around[i][0], y + self.circle_around[i][1]] < self.base_inten_min:
                self.status_around[i] = -1
            elif self.img[x + self.circle_around[i][0], y + self.circle_around[i][1]] > self.base_inten_max:
                self.status_around[i] = 1

        self.tmp = np.zeros(16, dtype='int8')
        self.max_length = 0
        for i in range(1, 16):
            if self.status_around[i] == 0:
                continue
            if self.status_around[i] == self.status_around[i - 1]:
                self.tmp[i] = self.tmp[i - 1] + 1
                if i == self.tmp[i]:
                    self.tmp[0] = i
            elif self.tmp[i - 1] > self.max_length:
                self.max_length = self.tmp[i - 1]

        if self.status_around[0] != 0 and self.status_around[0] == self.status_around[15]:
            self.max_length = max(self.max_length, self.tmp[0] + self.tmp[15])

        return self.max_length >= 12

    def doSuppression(self):
        pass

def draw_corners(img, corners):
    for (x, y) in corners:
        cv2.circle(img, (y, x), 10, (0, 255, 0))
    cv2.imshow('FAST corners', img)
    cv2.waitKey(0)

if __name__ == '__main__':

    img = cv2.imread('/home/tung/Downloads/shapes.jpg')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

    fast = FAST()
    corners = fast.detect_corners(img)

    print 'There are {} corners in the image'.format(len(corners))

    draw_corners(img, corners)