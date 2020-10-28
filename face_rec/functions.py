

import pywt
import  numpy as np

import cv2

from django.conf import settings









def predicting_images_functions(img_path):


    baseUrl = settings.BASE_DIR_ROOT + settings.STATIC_URL



    lol = []
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        baseUrl+'ML/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(baseUrl+
        'ML/haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    x, y, w, h = faces[0]

    for (x, y, w, h) in faces:
        face_img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = face_img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cropped_img = np.array(roi_color)

    def w2d(img, mode='haar', level=1):
        imArray = img
        # Datatype conversions
        # convert to grayscale
        imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
        # convert to float
        imArray = np.float32(imArray)
        imArray /= 255;
        # compute coefficients
        coeffs = pywt.wavedec2(imArray, mode, level=level)

        # Process Coefficients
        coeffs_H = list(coeffs)
        coeffs_H[0] *= 0;

        # reconstruction
        imArray_H = pywt.waverec2(coeffs_H, mode);
        imArray_H *= 255;
        imArray_H = np.uint8(imArray_H)

        return imArray_H

    scalled_raw_img = cv2.resize(cropped_img, (32, 32))
    img_har = w2d(cropped_img, 'db1', 5)

    scalled_img_har = cv2.resize(img_har, (32, 32))
    combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32,
                                                                                               1)))  # stacking the vectorizede img and raw img one over the other that is the x
    lol.append(combined_img)
    lol = np.array(lol).reshape(len(lol), 4096).astype(float)  # so it will be one row and 4097 columns

    return lol