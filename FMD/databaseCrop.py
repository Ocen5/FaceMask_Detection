import cv2
import dlib
import numpy as np
from imutils import paths

maskImages = paths.list_images("D:/Users/Marco/Desktop/706736_1237011_bundle_archive/Mask/Mask")
noMaskImages = paths.list_images("D:/Users/Marco/Desktop/Face mask detection/2.Progetto/face-mask-detector/face-mask-detector/dataset/without_mask/")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/Users/Marco/Desktop/shape_predictor_68_face_landmarks.dat")


# creazione nuovo dataset a partire dall'originale, con immagini esclusivamente croppate sul volto
# creazione cropped_without_mask
i = 0
for image in noMaskImages:
    print(image)
    find = False
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    rects = detector(img, 1)
    print(rects)
    for (a, rect) in enumerate(rects):
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()

        if x > 0 and y > 0:
            crop = img[y:h, x:w]
            fileName = "D:/Users/Marco/Desktop/Face mask detection/2.Progetto/face-mask-detector/face-mask-detector/dataset/cropped/cropped_without_mask/" + str(i) + "_" + str(a) + ".jpg"
            cv2.imwrite(fileName, crop)

        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        shape = predictor(img, rect)
        coords = np.zeros((68, 2), dtype="int")
        for j in range(0, 68):
            coords[j] = (shape.part(j).x, shape.part(j).y)
        shape = coords
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        find = True
    if find:
        fileName = "D:/Users/Marco/Desktop/Face mask detection/2.Progetto/face-mask-detector/face-mask-detector/dataset/pointed_images/noMask/00" + str(i) + "_" + str(a) + ".jpg"
        cv2.imwrite(fileName, img)
    i = i + 1


# creazione cropped_with_mask
i = 0
for image in maskImages:
    print(image)
    find = False
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    rects = detector(img, 1)
    print(rects)
    for (a, rect) in enumerate(rects):
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()

        if x > 0 and y > 0:
            crop = img[y:h, x:w]
            fileName = "D:/Users/Marco/Desktop/706736_1237011_bundle_archive/Mask/crop_mask/" + str(i) + "_" + str(a) + ".jpg"
            cv2.imwrite(fileName, crop)

        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        shape = predictor(img, rect)
        coords = np.zeros((68, 2), dtype="int")
        for j in range(0, 68):
            coords[j] = (shape.part(j).x, shape.part(j).y)
        shape = coords
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        find = True
    if find:
        fileName = "D:/Users/Marco/Desktop/706736_1237011_bundle_archive/Mask/pointed_img/00" + str(i) + ".jpg"
        cv2.imwrite(fileName, img)
    i = i + 1
