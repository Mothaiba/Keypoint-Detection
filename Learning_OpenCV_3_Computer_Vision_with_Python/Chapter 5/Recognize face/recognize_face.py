'''
This program has 2 actions:
    First is, train and recognize face obtained from webcam
    Second is, test the multiprocessing library, run parallel
'''

import cv2
import numpy as np
from os import listdir
from os.path import join, isfile
import datetime
from multiprocessing import Pool
from contextlib import closing

'''
This function takes input (directory that contains the face-images, the index
of the person who provides those faces (default: 0))
Output is a tuple: (array of matrices represent those face, array of
person-indexes)
'''
def loadFaces(_path, idx=0):
    _images = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]
    print 'len(_images):', len(_images)

    images = []
    people = []

    for _image in _images:
        image = cv2.imread(_image, 0)
        if image is not None and not np.array_equal(image.shape, [0, 0]):
            images.append(cv2.resize(image, (200, 200)))
            people.append(idx)

    return np.array(images, dtype='uint8'), np.array(people, dtype=int)

def recognize_face_in_image(_image):
    global model

    image = cv2.imread(_image, 0)
    person, confidence = model.predict(image)

def test_multicores(_images):
    last_time = datetime.datetime.now()
    for _image in _images:
        recognize_face_in_image(_image)
    print 'single core time:', datetime.datetime.now() - last_time

    last_time = datetime.datetime.now()
    with closing(Pool()) as pool:
        pool.map(recognize_face_in_image, _images)
        pool.terminate()
    print 'multicores time:', datetime.datetime.now() - last_time

if __name__ == '__main__':
    _path = './Tung_face/'

    images, people = loadFaces(_path)

    model = cv2.face.createEigenFaceRecognizer()
    model.train(images, people)

    _images = [join(_path, f) for f in listdir(_path) if isfile(join(_path, f))]

    # test_multicores(_images)

    face_detector = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_alt2.xml')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        frame = np.fliplr(frame).astype('uint8')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_detector.detectMultiScale(gray_frame, 1.1, 5)

        for face in faces:
            x, y, w, h = face
            image = cv2.resize(frame[x:x + w, y:y + h], (200, 200))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            person, confidence = model.predict(image)
            print 'confidence:', confidence

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))
            cv2.putText(frame, str(confidence), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow('Face recognizer', frame)

        if cv2.waitKey(1000/12) == ord('q'):
            break

    cv2.destroyAllWindows()

