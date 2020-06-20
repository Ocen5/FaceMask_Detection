# import delle librerie necessarie
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
import tkinter.filedialog as tkFileDialog
import tkinter.messagebox

path = ""
imgOriginalScene = ""
fimage = ""
finalimage = ""
h = 0
w = 0
faceNet = ""
maskNet = ""
vs = ""
feedback = ""
imagecam = ""
frame = ""


# ----------------------------------------------- DETECT FROM IMAGE -------------------------------------------------
def select_image():
    # carico l'immagine da percorso, la assegno alla label per visualizzarla e prelevo le sue dimensioni
    global path, image, imgOriginalScene, h, w
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        print(path)
        imgOriginalScene = cv2.imread(path)

        image = cv2.resize(imgOriginalScene, (400, 300))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        Label_6.configure(image=image)
        (h, w) = imgOriginalScene.shape[:2]


def predict_face():
    # se il path è vuoto significa che non è stata scelta nessuna immagine e quindi visualizzo il warning
    if path == "":
        tkinter.messagebox.showwarning("IMMAGINE NON SELEZIONATA",
                                       "Per favore seleziona l'immagine \nper effettuare il riconoscimento \ndella mascherina")
    else:
        # caricare il face detector
        Label_7.configure(text="Caricando il modello di face detector...\n")
        prototxtPath = "face_detector/deploy.prototxt"
        weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        net = cv2.dnn.readNet(prototxtPath, weightsPath)

        # caricamento modello addestrato
        text = Label_7.cget("text") + "Caricando il modello per il riconoscimento \ndella mascherina...\n"
        Label_7.configure(text=text)
        model = load_model("model.h5")

        # costruire blob dalle immagini
        blob = cv2.dnn.blobFromImage(imgOriginalScene, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # passare il blob, per ottenere la face detection
        text = Label_7.cget("text") + "Analisi del volto...\n"
        Label_7.configure(text=text)
        net.setInput(blob)
        detections = net.forward()

        # loop per il detect
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
                face = imgOriginalScene[startY:endY, startX:endX]
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
                cv2.putText(imgOriginalScene, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(imgOriginalScene, (startX, startY), (endX, endY), color, 2)

        # chiamata alla funzione per settare l'immagine nella label corrispondente
        set_image(imgOriginalScene)


def set_image(fimage):
    global finalimage
    # serie di operazioni per fare visualizzare l'immagine elaborata nella label
    finalimage = cv2.resize(fimage, (400, 300))
    finalimage = cv2.cvtColor(finalimage, cv2.COLOR_BGR2RGB)
    finalimage = Image.fromarray(finalimage)
    finalimage = ImageTk.PhotoImage(finalimage)
    Label_8 = Label(tab1, borderwidth=2, relief="solid", image=finalimage, height=300, width=400)
    Label_8.grid(row=5, column=2, rowspan=2)


def photo_img():
    if imgOriginalScene == "":
        tkinter.messagebox.showwarning("IMMAGINE NON SELEZIONATA",
                                       "Per favore seleziona l'immagine \nper effettuare il riconoscimento \ndella mascherina")
    else:
        cv2.imwrite("imgs/photoimg/img_detected.jpg", imgOriginalScene)
        text = Label_7.cget("text") + "Immagine salvata correttamente in: imgs/photoimg/\n"
        Label_7.configure(text=text)


# ----------------------------------------------- DETECT FROM VIDEO -------------------------------------------------
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

        # non considerare immagini con confidence inferiore a 0.20
        if confidence > 0.20:
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


def init_cam():
    global vs, faceNet, maskNet
    # caricare il face detector
    Label_6_2.configure(text="Caricando il modello di face detector...\n")
    prototxtPath = "face_detector/deploy.prototxt"
    weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # caricamento modello addestrato
    text = Label_6_2.cget("text") + "Caricando il modello per il riconoscimento \ndella mascherina...\n"
    Label_6_2.configure(text=text)
    maskNet = load_model("model.h5")

    # inizializza il video capture per far partire la cam
    text = Label_6_2.cget("text") + "Aprendo la videocamera...\n"
    Label_6_2.configure(text=text)
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)
    face_recognition()


def face_recognition():
    global feedback, frame
    # estraggo il frame da video capture (tramite read()) e lo ridimensiono in base
    # alle dimensioni della label dove sarà visualizzato
    feedback, frame = vs.read()
    cv2.resize(frame, (400, 300))

    # rilevo il volto in questo singolo frame e determino se indossa la maschera o no
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

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
    set_image_cam(frame)


def set_image_cam(frm):
    global imagecam
    # serie di operazioni per fare visualizzare l'immagine elaborata nella label
    imagecam = cv2.resize(frm, (400, 300))
    imagecam = cv2.cvtColor(imagecam, cv2.COLOR_BGR2RGB)
    imagecam = Image.fromarray(imagecam)
    imagecam = ImageTk.PhotoImage(imagecam)
    Label_5_2.configure(image=imagecam)
    Label_5_2.after(10, face_recognition)


def photo_vid():
    global frame
    if imagecam == "":
        tkinter.messagebox.showwarning("ERRORE REALTIME VIDEO",
                                       "Si è verificato un errore \nper effettuare il riconoscimento \ndalla videocamera")
    else:
        cv2.imwrite("imgs/photovid/video_detected.jpg", frame)
        text = Label_6_2.cget("text") + "immagine salvata correttamente in: \nimgs/photovid/\n"
        Label_6_2.configure(text=text)


def close():
    global feedback, vs
    # chiusura finestra GUI
    if feedback:
        vs.release()
        cv2.destroyAllWindows()
    root.destroy()


root = Tk()  # creazione finestra GUI
root.title("Mask Detection")  # titolo finestra GUI
root.iconbitmap("imgs/py.ico")  # icona finestra GUI
ws = root.winfo_screenwidth()  # larghezza dello schermo
hs = root.winfo_screenheight()  # altezza dello schermo
root_width = 1366  # larghezza della finestra GUI
root_heigth = 768  # altezza della finestra GUI
# metodo per far visualizzare la finestra GUI al centro dello schermo indipendentemente dalle sue risoluzioni
root.geometry(
    "%dx%d+%d+%d" % (root_width, root_heigth - 70, (ws / 2) - ((root_width + 16) / 2), (hs / 2) - (root_heigth / 2)))
root.resizable(False, False)  # finestra GUI non ridimensionabile

# operazioni di stile: settare lo sfondo bianco e il carattere di testo
customed_style = ttk.Style()
customed_style.configure('Custom.TNotebook.Tab', font=('arial', 10, 'bold'), foreground='white')

# creo il tab e lo assegno alla finestra GUI e modifico font e immagini
tab_control = ttk.Notebook(root, style='Custom.TNotebook')

tab1 = Frame(tab_control, bg='white')
tab2 = Frame(tab_control, bg='white')

phototabcam = ImageTk.PhotoImage(file="imgs/tabcam.png")
phototabimg = ImageTk.PhotoImage(file="imgs/tabimg.png")
tab_control.add(tab1, text='RILEVA DA IMMAGINE', image=phototabimg, compound="center")
tab_control.add(tab2, text='RILEVA DA VIDEO', image=phototabcam, compound="center")

# ---------------------------------------------- tab immagine GUI ------------------------------------------------
# label contenente l'immagine al top
phototop = ImageTk.PhotoImage(file="imgs/top.png")
Label_1 = Label(tab1, font=('arial', 18, 'bold'), image=phototop, bg='white', justify="center")
Label_1.grid(row=0, column=0, columnspan=3)

# bottone di caricamento dell'immagine da file
photobtn = ImageTk.PhotoImage(file="imgs/btn.png")
Label_2 = Button(tab1, font=('arial', 10, 'bold'), image=photobtn, text="CARICA IMMAGINE", compound="center",
                 activebackground='white', borderwidth=0, bg='white', fg="white", command=select_image)
Label_2.grid(row=3, column=0)

# bottone di avvio processo di rilevamento
Label_3 = Button(tab1, font=('arial', 10, 'bold'), image=photobtn, text="RILEVA MASCHERINA", compound="center",
                 activebackground='white', borderwidth=0, bg='white', fg="white", command=predict_face)
Label_3.grid(row=3, column=2, padx=(0, 220))

# bottone salva immagine rilevata
Label_4 = Button(tab1, font=('arial', 10, 'bold'), image=photobtn, text="SALVA IMMAGINE", compound="center",
                 activebackground='white', borderwidth=0, bg='white', fg="white", command=photo_img)
Label_4.grid(row=3, column=2, padx=(220, 0))

# label di spazio
Label_5 = Label(tab1, height=1, bg='white')
Label_5.grid(row=4, column=0, columnspan=3)

# label per l'immagine selezionata
photoico = ImageTk.PhotoImage(file="imgs/img.png")
Label_6 = Label(tab1, borderwidth=2, relief="solid", image=photoico, height=300, width=400)
Label_6.grid(row=5, column=0, rowspan=2)

# label per la cronologia delle operazioni
empty = ImageTk.PhotoImage(file="imgs/empty.png")
Label_7 = Label(tab1, font=('arial', 12, 'bold'), borderwidth=2, image=empty, relief="solid", compound="center",
                height=150, width=400)
Label_7.grid(row=5, column=1)

# bottone di chiusura
Label_9 = Button(tab1, font=('arial', 10, 'bold'), image=photobtn, text="CHIUDI", compound="center",
                 activebackground='white', borderwidth=0, bg='white', fg="white", command=close)
Label_9.grid(row=6, column=1)

# label per l'immagine in bottom
photobottom = ImageTk.PhotoImage(file="imgs/bottom.png")
Label_10 = Label(tab1, font=('arial', 8, 'bold'),
                 text="Realized by:\n Belvedere Vincenzo\n Conticchio Giuseppe, Leone Marco", pady=10,
                 image=photobottom,
                 fg="white", compound="center", bg='white', justify="center")
Label_10.grid(row=7, column=0, columnspan=3)

# ---------------------------------------------- tab video GUI ------------------------------------------------
# label contenente l'immagine al top
Label_1_2 = Label(tab2, font=('arial', 18, 'bold'), image=phototop, bg='white', justify="center")
Label_1_2.grid(row=0, column=0, columnspan=2)

# bottone di caricamento del videocapture
Label_2_2 = Button(tab2, font=('arial', 10, 'bold'), image=photobtn, text="AVVIA STREAMING", compound="center",
                   activebackground='white', borderwidth=0, bg='white', fg="white", command=init_cam)
Label_2_2.grid(row=3, column=0, padx=(0, 220))

# bottone salva immagine da videocapture
Label_3_2 = Button(tab2, font=('arial', 10, 'bold'), image=photobtn, text="SALVA IMMAGINE", compound="center",
                   activebackground='white', borderwidth=0, bg='white', fg="white", command=photo_vid)
Label_3_2.grid(row=3, column=0, padx=(220, 0))

# label di spazio
Label_4_2 = Label(tab2, height=1, bg='white')
Label_4_2.grid(row=4, column=0, columnspan=2)

# label videocapture
camico = ImageTk.PhotoImage(file="imgs/cam.png")
Label_5_2 = Label(tab2, borderwidth=2, relief="solid", image=camico, compound="center", height=300, width=400)
Label_5_2.grid(row=5, column=0, rowspan=2)

# label per la cronologia delle operazioni
Label_6_2 = Label(tab2, font=('arial', 12, 'bold'), borderwidth=2, relief="solid", image=empty, compound="center",
                  height=150, width=400)
Label_6_2.grid(row=5, column=1)

# bottone di chiusura
Label_7_2 = Button(tab2, font=('arial', 10, 'bold'), image=photobtn, text="CHIUDI", compound="center",
                   activebackground='white', borderwidth=0, bg='white', fg="white", command=close)
Label_7_2.grid(row=6, column=1)

# label per l'immagine in bottom
Label_8_2 = Label(tab2, font=('arial', 8, 'bold'),
                  text="Realized by:\n Belvedere Vincenzo\n Conticchio Giuseppe, Leone Marco", pady=10,
                  image=photobottom,
                  fg="white", compound="center", bg='white', justify="center")
Label_8_2.grid(row=7, column=0, columnspan=2)

# espande e riempie la finestra GUI con i contenuti in tab_control
tab_control.pack(expand=1, fill='both')

# esegui root (finestra GUI)
root.mainloop()
