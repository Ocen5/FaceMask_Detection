from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2


# caricare il face detector
print("[INFO] loading face detector model...")
prototxtPath = "D:/Users/Marco/Desktop/Face mask detection/2.Progetto/face-mask-detector/face-mask-detector/face_detector/deploy.prototxt"
weightsPath = "D:/Users/Marco/Desktop/Face mask detection/2.Progetto/face-mask-detector/face-mask-detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxtPath, weightsPath)


# caricamento modello addestrato
print("[INFO] loading face mask detector model...")
model = load_model("C:/Users/Marco/PycharmProjects/FMD2/model.h5")


# inserire il percorso dell'immagine da analizzare
image = cv2.imread("D:/Users/Marco/Desktop/prova imm2.jpg")
orig = image.copy()
(h, w) = image.shape[:2]


# costruire blob dalle immagini
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))


# passare il blob, ottenere face detection
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()


# loop per il detect
for i in range(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with
	# the detection
	confidence = detections[0, 0, i, 2]

	# non considerare immagini con confidence inferiore a 0.60
	if confidence > 0.60:
		# calcolo delle coordinate (x, y) del riquadro di delimitazione dell'oggetto
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		#assicurarsi delle dimensione dei rettangoli
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		#estrarre la roi del viso, convertire BGR in RGB, resize a 224x224 e preprocessing
		face = image[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# predire la presenza o meno della mascherina
		(mask, withoutMask) = model.predict(face)[0]

		# determinare l'etichetta della classe e il colore del rettangolo
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# includere la probabilità nel modello
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# visualizza l'etichetta e il rettangolo
		cv2.putText(image, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)


# mostrare l'output
cv2.imshow("Output", image)
cv2.waitKey(0)