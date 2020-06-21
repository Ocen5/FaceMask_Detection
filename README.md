## 									FACEMASK DETECTION

#### **INTRODUZIONE**

Il primo task riguardo il progetto di Sistemi ad agenti che stiamo portando avanti, si basa sul riconoscimento della mascherina. In questo periodo di forte crisi dovuto alla presenza del COVID-19, abbiamo pensato, insieme alla prof De Carolis, di realizzare un modello che potesse essere utile nel riconoscimento della mascherina sul volto della gente, sia in Real time, che “staticamente” (analizzando la singola immagine).

#### **IDEE DI PROGETTO**

L’idea che abbiamo utilizzato per portare a termine tale task, è stato quello di utilizzare una delle tecniche più conosciute nell’ambito dell’AI, l’uso di una Neural Network. Il primo step è stato quello di approfondire tale tematica, in maniera piuttosto teorica, aiutandoci sia con l’utilizzo di libri, articoli presenti sul web e consultando la documentazione delle diverse librerie che ci sarebbero servite.

#### **ORGANIZZAZIONE DEL PROGETTO**

#### **<u>FASE DI TRAIN:</u>**

![1](https://user-images.githubusercontent.com/48212689/85221118-934b7280-b3b1-11ea-9859-aec9cda6631f.png)

- **<u>DATASET:</u>**

https://drive.google.com/drive/folders/1JtzDcqpgAuh15PMcoc6UHQEqT_SEMT8J?usp=sharing

**ESEMPI:**

![0](https://user-images.githubusercontent.com/48212689/85221547-c04d5480-b3b4-11ea-8bca-edfcaca49c59.jpg)													![0-with-mask](https://user-images.githubusercontent.com/48212689/85221572-effc5c80-b3b4-11ea-903f-df4bca158f10.jpg)



- <u>**PREPROCESSING:**</u> tale fase si è basata sull’processing delle immagini presenti nel dataset.
  Esso è utile farlo, al fine di addestrare la rete esclusivamente sulla parte dell’immagine
  utile a noi (il viso). Per fare ciò abbiamo seguito una metodologia consigliataci dal Dr.
  Nicola Macchiarulo; tale metodologia consiste nel:
  		**•**   Individuare il volto attraverso l’uso di un **Face Detector** (abbiamo utilizzato la libreria dlib)
  		**•**   Una volta individuato il volto, abbiamo effettuato il crop esclusivamente sulla zona individuata 			dal precedente modello
  		**•**   Questo ci ha portato a creare un nuovo dataset composto da due sottocartelle                                        			(cropped_with_mask, cropped_without_mask) :
  			https://drive.google.com/drive/folders/1JtzDcqpgAuh15PMcoc6UHQEqT_SEMT8J
  		**•**   A causa della presenza di immagine duplicate all’interno del dataset, abbiamo utilizzato un 			programma: “Similar Images”. Questo ha fatto si che il dataset diventasse sbilanciato, 					                                           			(cropped_with_mask<cropped_without_mask). Per risolvere tale problema, sono state aggiunte 			altre immagini, opportunamente croppate, nella sottocartella di nostro interesse                                                                              			(cropped_with_mask).

  ![2](https://user-images.githubusercontent.com/48212689/85221424-fdfdad80-b3b3-11ea-9822-ca3d0fb18748.png)

  ![3](https://user-images.githubusercontent.com/48212689/85221514-8d0ac580-b3b4-11ea-8df7-aa433c0e85ea.png)

  

- **<u>SPLIT:</u>** al fine di addestrare la rete (MobileNetV2), abbiamo suddiviso il dataset in
  **•** **train ➔ 80%**
  **•** **validation ➔ 10%**
  **•** **test ➔ 10 %**

- <u>**TRAIN:**</u> abbiamo effettuato il training basandoci su N_epochs (numero di epoche)= 20. Abbiamo fatto fine tuning della rete menzionata precedentemente

  #### **<u>FASE DI TEST:</u>**

  ![4](https://user-images.githubusercontent.com/48212689/85221660-e1fb0b80-b3b5-11ea-8236-7036c75185f6.png)

  

  **•**   Dopo aver effettuato il test, abbiamo definito l’accuratezza nella predizione. I valori sono riportati in       	basso nella sezione: “metriche utilizzate”. Ci siamo serviti anche di una Confusion matrix(presente in    	basso) utile a valutare gli errori commessi e non nella fase di predizione. Infine il modello viene   	 	serializzato sottoforma di “file.h5”

  #### <u>**FASE APPLICATIVA:**</u>

  ![5](https://user-images.githubusercontent.com/48212689/85222073-07d5df80-b3b9-11ea-9ffe-70f352c84ebe.png)

![6](https://user-images.githubusercontent.com/48212689/85222183-803ca080-b3b9-11ea-9b8f-9225f826ffdb.png)

**•**  Al termine del tutto ci siamo concentrati sull’utilizzo del modello creato, affinché riconoscesse la presenza 	o meno della mascherina da immagini esterne scelte su internet, per poi effettuare un detect in tempo 	reale (mediante l’utilizzo della webcam del pc) sfruttando le funzionalità della libreria OpenCV.
**•**  **Interfaccia Grafica** ➔ al fine di utilizzare tale progetto in maniera molto più rapida ed intuitiva, abbiamo 	considerato la creazione di una semplice interfaccia grafica che permette di effettuare le operazioni di 	  	riconoscimento di mascherina sul volto sia da immagine che da videocamera con la possibilità di salvare   	l'output come mostrato di seguito:

![g1](https://user-images.githubusercontent.com/48212689/85222477-9cd9d800-b3bb-11ea-88a4-864a692104f6.png)



![g2](https://user-images.githubusercontent.com/48212689/85222476-9c414180-b3bb-11ea-992b-f3ebbde10e6b.png)

#### <u>**TRAINING & VALIDATION ACCURACY**</u>:

![plot](https://user-images.githubusercontent.com/48212689/85222500-c72b9580-b3bb-11ea-82a0-f18764d61cac.png)

#### <u>**TRAINING & VALIDATION LOSS:**</u>

![plot1](https://user-images.githubusercontent.com/48212689/85222499-c692ff00-b3bb-11ea-8fbb-dd664cec6676.png)

#### <u>**CONFUSION MATRIX:**</u>

![8](https://user-images.githubusercontent.com/48212689/85222705-c136b400-b3bd-11ea-80e7-b7fd7a395c30.png)

#### <u>**METRICHE UTILIZZATE:**</u>

![9](https://user-images.githubusercontent.com/48212689/85222803-4b7f1800-b3be-11ea-80f2-6b57bf09a248.png)
