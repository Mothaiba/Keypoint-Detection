import cv2
import numpy as np

class BRIEF:

    def __init__(self, img, a_size = 9, descriptor_length = 512):
        # save some initial information
        if len(img.shape) > 2 or img.shape[2] > 1:
            self.img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            self.img = img
        self.rows, self.cols = img.shape[:2]
        self.a_size = a_size
        self.descriptor_length = descriptor_length

        # uniformly create a list of pixel-pairs to compare
        self.x1 = np.random.random_integers(- int(a_size / 2), int(a_size / 2), self.descriptor_length)
        self.x2 = np.random.random_integers(- int(a_size / 2), int(a_size / 2), self.descriptor_length)
        self.y1 = np.random.random_integers(- int(a_size / 2), int(a_size / 2), self.descriptor_length)
        self.y2 = np.random.random_integers(- int(a_size / 2), int(a_size / 2), self.descriptor_length)

        self.comparative_pairs = np.vstack((self.x1.T, self.y1.T, self.x2.T, self.y2.T)).T

        # free memory
        del self.x1
        del self.x2
        del self.y1
        del self.y2

    # get descriptor for pixel at position (x, y) of the pre-defined image
    def getDescriptor(self, x, y):
        # array which stores the descriptor
        descriptor = np.zeros(self.descriptor_length, dtype=bool)

        # compute the descriptor
        for i in range(len(self.comparative_pairs)):
            (x1, y1, x2, y2) = self.comparative_pairs[i]
            p1_x = x + x1
            p1_y = y + y1
            p2_x = x + x2
            p2_y = y + y2

            # if p1_x < 0 or p2_x < 0 or p1_y < 0 or p2_y < 0 or\
            # p1_x >= self.rows or p2_x >= self.rows or p1_y >= self.cols or p2_y >= self.cols:
            #     continue
            try:
                if self.img[p1_x, p1_y] > self.img[p2_x, p2_y]:
                    descriptor[i] = True
            except:
                pass

        return descriptor

    # calculate the distance between 2 descriptors
    # the larger the value, the more different the 2 descriptors are
    def compareDescriptor(self, desc1, desc2):
        distance = np.sum(desc1 != desc2)
        return distance


if __name__ == '__main__':
    import FAST_corner_detector as FAST

