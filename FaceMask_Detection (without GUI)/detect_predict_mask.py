import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


def detect_and_predict_mask(frame, faceNet, maskNet):
	# prelevo le dimensioni del frame video e poi costruisco un blob su di esso
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# do in input il blob alla rete per ottenere il rilevamento facciale
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# inizializzo la lista dei volti, le loro posizioni, e la lista delle predizioni fatte dalla rete
	faces = []
	locs = []
	preds = []

	# ciclo su ciò che è stato rilevato
	for i in range(0, detections.shape[2]):
		# estraggo la confidence (valore di probabiltà) associata al riconoscimento
		confidence = detections[0, 0, i, 2]

		# non considerare immagini con confidence inferiore a 0.60
		if confidence > 0.60:
			# calcolo delle coordinate (x, y) del riquadro di delimitazione dell'oggetto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# assicurarsi delle dimensione dei rettangoli
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# estrarre la roi del volto, convertire BGR in RGB, resize a 224x224 e preprocessing
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# aggiunta del box di rilevamento facciale al volto
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# controllo se il volto è stato scansionato verificando la lunghezza della lista
	if len(faces) > 0:
		# per velocizzare il procedimento facciamo le predizioni su tutta la lista
		# rispetto al ciclo precedente che analizzava frame per frame
		preds = maskNet.predict(faces)

	# restituisco 2 liste: locs che contiene le posizioni del volto (4 vertici del rettangolo facciale)  e
	# preds che contiene le predizioni sulle mascherine per i volti scansionati
	return (locs, preds)