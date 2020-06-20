from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2
import detect_predict_mask


# caricare il face detector
print("[INFO] loading face detector model...")
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# caricamento modello addestrato
print("[INFO] loading face mask detector model...")
maskNet = load_model("model.h5")

# inizializza il video stream per far partire la cam
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop sui frames di video stream
while True:
	# estraggo il frame da video stream (tramite read()) e lo ridimensiono in base
	# alle dimensioni della label dove sarà visualizzato
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# rilevo il volto in questo singolo frame e determino se indossa la maschera o no
	(locs, preds) = detect_predict_mask.detect_and_predict_mask(frame, faceNet, maskNet)

	# ciclo sui volti rilevati e le loro posizioni
	for (box, pred) in zip(locs, preds):
		# estraggo le posizioni del volto (vertici rettangolo) e le predizioni
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determinare l'etichetta della classe e il colore del rettangolo
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# includere la probabilità nel modello
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# visualizza l'etichetta e il rettangolo
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# mostra finestra con video
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# se premi q esci dal ciclo while
	if key == ord("q"):
		break

# chiude la finestra e ferma il videostream
cv2.destroyAllWindows()
vs.stop()