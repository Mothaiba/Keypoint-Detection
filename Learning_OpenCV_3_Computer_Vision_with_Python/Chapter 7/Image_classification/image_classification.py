'''
This program classify images (for now, only classify an image is car or non-car
using Bag-of-SURF
with SVM as the classifier
'''

import cv2
import numpy as np

import imutils

from os.path import join, isfile
from os import listdir

LABEL_TO_INT = {}
INT_TO_LABEL = {}

def trainFunc(_train_path):
    _categories = listdir(_train_path)

    detect = cv2.xfeatures2d.SURF_create()
    extract = cv2.xfeatures2d.SURF_create()

    flann_params = dict(algorithm=1, tree=5)
    flann = cv2.FlannBasedMatcher(indexParams=flann_params, searchParams={})

    def extract_features(_image):
        image = imutils.resize(cv2.imread(_image, 0), height=400)
        return extract.compute(image, detect.detect(image))[1]

    def bow_of(_image):
        image = imutils.resize(cv2.imread(_image, 0), height=400)
        return bow_extractor.compute(image, detect.detect(image))

    bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
    bow_extractor = cv2.BOWImgDescriptorExtractor(extract, flann)

    for each in _categories:

        if not LABEL_TO_INT.has_key(each):
            LABEL_TO_INT[each] = len(INT_TO_LABEL)
            INT_TO_LABEL[len(INT_TO_LABEL)] = each

        _path = join(_train_path, each)
        _images = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]

        for _image in _images[:5]:
            bow_kmeans_trainer.add(extract_features(_image))

    vocabulary = bow_kmeans_trainer.cluster()
    bow_extractor.setVocabulary(vocabulary)

    train_data = []
    train_label = []

    for each in _categories:
        _path = join(_train_path, each)
        _images = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]

        for _image in _images[:5]:
            features = bow_of(_image)
            train_data.extend(features)
            train_label.append(LABEL_TO_INT[each])

    svm = cv2.ml.SVM_create()
    svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))

    return svm, bow_of


def predictFunc(model, bow_of, _test_img):
    bow = bow_of(_test_img)
    result = model.predict(bow)[1][0][0]
    return INT_TO_LABEL[result]

if __name__ == '__main__':
    _train_path = './TrainImages/'
    svm, bow_of = trainFunc(_train_path)

    _test_path = './TestImages/'
    _test_imgs = [join(_test_path, f) for f in listdir(_test_path) if isfile(join(_test_path, f))]

    for idx, _test_img in enumerate(_test_imgs[:5]):
        category = predictFunc(svm, bow_of, _test_img=_test_img)
        print category
        test_img = imutils.resize(cv2.imread(_test_img, 1), height=400)
        cv2.putText(test_img, category, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 0), cv2.LINE_4)
        cv2.imshow(str(idx), test_img)

    cv2.waitKey()
