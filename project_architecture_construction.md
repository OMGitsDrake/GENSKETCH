## 1. Dataset

1. Scaricare immagini dal repo di Sketchy
2. impostare normalizzazione quando necessaria, non a monte del procedimento di addestramento o inferenza

## 2. Inferenza

### 2.1 Classification Head
Deve imparare a classificare i disegni nel modo corretto, per passare la classe alla **VAE Head**

### 2.2 VAE Head
Riceve la classe predetta dalla **Classification Head** insieme allo sketch, e deve generare un'immagine in scala di grigi che rappresenta un disegno completo e dettagliato.

## 3. Addestramento

### 3.1 Classification Head
Classificazione tramite **MLP**: riconoscimento della classe dello sketch in ingresso.
- loss fn: `Cross entropy`

### 3.2 VAE Pretraining
Il VAE va pre-addestrato per riuscire a generare immagini di disegni completi (sempre in scala di grigi) a partire dalla label e dallo sketch, senza che il classificatore intervenga.

In questo modo il VAE saprà generare un'immagine di un diseno completo a partire dalla label in uscita dal classificatore e da rumore bianco

### 3.3 VAE Head
Generazione di immagini a partire da rumore bianco `~N(0,1)`, lo sketch iniziale e la label predetta dal classificatore
- loss_fn: `Binary cross entropy`