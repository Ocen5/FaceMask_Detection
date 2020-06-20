# FaceMask_Detection

## Dataset 

###  https://drive.google.com/drive/folders/1JtzDcqpgAuh15PMcoc6UHQEqT_SEMT8J?usp=sharing



## SCHEMA INTRODUTTIVO

### TRAIN: PREPROCESSING -> TRAIN -> SALVATAGGIO MODELLO

 ### TEST: CARICAMENTO DEL MODELLO SALVATO-> PREDIZIONE (IN REAL TIME/IMMAGINE) -> RISULTATO 

## Preprocessing delle immagini del dataset

### Se ci sarà bisogno, avevamo pensato di effetturare un preprocessing delle immagini presenti nel dataset, in modo da rendere equilibrato il tutto. Primo passo della pipeline utile a soddisfare questo task

## Addestramento

### L'idea è quella di addestrare un modello (Rete Neurale  Convoluzionale), che riesca ad apprendere e dunque a disambiguare la presenza della mascherina sul volto di un individuo.

### Abbiamo pensato di creare un modulo nel quale andremo a fare il Train della rete, andando ad acquisire le immagini presenti nel dataset (caricato in precedenza). 

### Creazione del modello, e settaggio degli  hyperparameters della rete (pensavamo di utilizzare una VGG16, dato che ci è stata spiegata da lei)

### Utilizzare grafici di supporto per valutare l'andamento dell'addestramento, in modo da avere una configurazione migliore del modello

### Salvare il modello.

## Predizione

### Se l'addestramento va a buon fine, e i valori di accuratezza sono abbastanza alti, procedere con il task predizione

### Real Time o caricamento di un'immagine: abbiamo intenzione sia di effetturare il riconoscimento in real time attraverso l'acquisizione dello stream video dalla webcam oppure attraverso il caricamento di un immagine e l'analisi "statica" della stessa

## Interfaccia Grafica

### Creare una semplice interfaccia, in modo tale da avviare il tutto, evitando di accedere al codice e runnarlo





### 





### 



