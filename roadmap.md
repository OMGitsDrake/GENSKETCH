# 0 Creare il dataset
Ogni categoria ha 100 foto e un numero variabile di sketch (sempre maggiore di 100).

Organizzare le immagini come segue:
- per ogni foto che appartiene a un fold
- tutti i suoi sketch appartengono allo stesso fold

## Immagini per training e validation
Per ogni categoria, 60 foto sono per il training e 20 per la validation.
Le restanti 20 foto sono utilizzate come test set.
Gli sketch relativi a un'immagine la seguono nel suo fold.

- Definire una mappa fissa delle 125 categorie: nome -> ID (0–124).
- Iniziare con 10 categorie miste.
- Associare ogni sketch alla foto tramite l'ID nel nome del file.
- Assegnare a ciascuna foto uno dei 5 fold, separatamente per categoria.
- Creare un Dataset PyTorch che restituisca sketch, label, foto, photo_id e fold.
- Per il MLP, usare sketch e label, predisporre 125 output e valutare anche le prestazioni per categoria.
- Quando si aggiungono categorie, riprendere dal checkpoint e includere esempi delle categorie precedenti.

# 1 Addestrare il classificatore e salvare il modello
Immagini in ingresso di sketch e in uscita la label dell'oggetto rappresentato dall sketch

## Training
```
        Label
          |
          |--------------------------- 
          V                           V
Sketch -> MLP -> predicted label -> Loss
          ^                           |
          |                           |
          |                           |
           ------Backprop------------- 
                    (CE)
```

## Inference
```
IMG -> MLP -> predicted label
```

## Result
The `MLP` classifier is suitable for predicting the label of the sketch image, in order to obtain the correct category

# 2 Pre-adestrare il VAE
Una volta che il MLP sa produrre la categoia giusta patendo dall'immagine sketch, il VAE deve essere addestrato per produrre immagini complete e dettagliate a partire dallo sketch e dalla label di quest'ultimo.

## Training
```
  True Photo + Label                                             Sketch + Label     True Photo
            |                                                           |               |
            |                                                           |               |
            V                           (reparam. trick)                V               V
Sketch -> Encoder -> (mu, log_var) -> z = mu + sigma*eps ~ N(0,1) -> Decoder -> IMG -> Loss = Rec(IMG, True Photo) + B*KL
            ^                                                           ^               |
            |                                                           |               |
             ---------------------------------------------------------------------------
```

## Inference
```
Sketch + Label + z ~ N(0,1) -> Decoder -> IMG
```

## Result
`VAE` preaddestrato e in grado di costruire immagini dettagliate da uno sketch e dalla label della sua categoria

# 3 Valutazione performance VAE
Utilizzare il test set per vedere come si comporta il VAE con sketch e immagini mai viste

# 4 Assemblaggio e inferenza finale

L' `MLP` classifica lo sketch e passa la label predetta assieme allo sketch stesso al `decoder`, che assieme al valore `z` ricostruisce un'immagine dettagliata dello sketch, rendendolo "un disegno finito".
