import cv2
import numpy as np

def detect():
    face_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('../../haarcascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)

    while (True):
        ret, raw_frame = camera.read()
        frame = np.fliplr(raw_frame).astype('uint8')
        # print raw_frame.shape, frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40, 40))

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img, (ex + x, ey + y), (ex + x + ew, ey + y + eh), (0, 255, 0), 2)

        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 / 12) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  detect()