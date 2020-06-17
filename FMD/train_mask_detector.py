from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os
import confusion_matrix

# inizializzazione learning rate, n.epoche, batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32


# caricamento percorso dataset
print("[INFO] loading images...")
imagePaths = list(paths.list_images("D:/Users/Marco/Desktop/dat crop/cropped"))
data = []
labels = []


# loop in imagepaths
for imagePath in imagePaths:
	# estrazione label dal nome file
	label = imagePath.split(os.path.sep)[-2]

	# caricamento immagine (dimensione: 224x224) e preprocessing
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)

	# aggiornamento data e labels
	data.append(image)
	labels.append(label)


# conversione data e labels in due vettori numpy
data = np.array(data, dtype="float32")
labels = np.array(labels)


# Binarizzare le etichette
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)


# partizionamento del dataset in 80% train, 10% validation, 10%test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.1, stratify=labels, random_state=42)
(trainX, validX, trainY, validY) = train_test_split(trainX, trainY, test_size=0.1, random_state=41)


# utilizzo della classe ImageDataGenerator per aumentare la numerosità del set, considerando un'inclinazione della
# immagine di 20°
aug = ImageDataGenerator(
	rotation_range=20,
	fill_mode="nearest")


#caricamento della rete MobileNetV2, con i pesi ImageNet pre addestrati  e i layer del top non inclusi
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


# costruzione headModel, il quale sarà posizionato sul top del baseModel
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)


# posizionare l'headModel sul baseModel, model sarà il modello finale
model = Model(inputs=baseModel.input, outputs=headModel)


#congelare i pesi di tutti i layer del basemodel
for layer in baseModel.layers:
	layer.trainable = False


# compile model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])


# train della head della rete
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(validX, validY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


#predizine test set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)


#predizione con probabilità più alta
predIdxs = np.argmax(predIdxs, axis=1)


#mostrare classification_report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


#serializzazione modello in locale
print("[INFO] saving mask detector model...")
model.save("model.h5")


# plot training accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="valid_acc")
plt.title("Training and Validation accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")


# plot training loss
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="valid_loss")
plt.title("Training and Validation loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("plot1")

# creazione della matrice di confusione
confusion_matrix.show_confusion_matrix(testY.argmax(axis=1), predIdxs)
