23/06

Entrambi gli addestramenti fatti con 1 strato da 64 neuroni

Addestramenti con batch\_size=8 --> troppo piccolo e va in underfitting

Addestramento con batch\_size=32 --> overfitting dopo 10 epoche



Soluzioni all'overfitting:

&#x20;   1 Aumentato il vettore denso di output X

&#x20;   2 Aggiunta 2 layer

&#x20;       2.1 Aggiunta dropout a 0.5: no overfitting su traning ma loss troppo alta su validation X

&#x20;       2.2 Aumento numero di epoche X (peggio di 2.1)

&#x20;   3 Riduzione neuroni da 64:32 a 32:16, output\_dim a 16 X

&#x20;       3.1 Con 50 epoche

&#x20;   4 Riduzione ouput\_dim a 8 e vettore token a 70 X

&#x20;       4.1 Riduzione ad un solo layer con 32 neuroni



24/06

Rete RNN semplice non funziona, probabilmente il testo è troppo lungo.

Passiamo alle LSTM

1. LST con 32 neuroni e funzione di attivazione tanh (risultati migliori rispetto alla simpleRNN)

   1. aggiunta di dropout a 0.5 X
   2. Provato ad aumentare le epoche ma non diminuisce la val\_loss
2. Aggiungo un altro layer LST (Rete 32:16)

   1. All'aumentare delle epoche 30 --> 100 peggiora
   2. impostato mask\_zero=True e epoche 100 --> vicino all'overfitting
   3. Provo a introdurre la regolarizzazione L2

      1. con 30 epoche loss di training troppo alta --> provo ad aumentare a 70 epoche
      2. 70 epoche --> training meglio ma validation peggio
   4. Aggiungo batch normalization (con dropout)
   5. Tolgo dropout solo batchnorm --> peggio
3. Provo con rete 3 stack lstm (64:32:16)

   1. Con batchnorm, l2 e dropout --> sembra meglio, provo con 100 epoche
   2. Provo con 200/500 epoche su colab --> loss buona sul training ma male sul validation
   3. Provo con adagrad come ottimizzatore (200 epoche)



25/06

Il problema non è relativo al modello ma al dataset, ho pochi esempi e quindi andando ad aumentare la rete aumentavo solamente l'overfitting

Provo con rete GRU che sono più leggere

1. Rete gru con 48 neuroni, layer bidirezionale, batchnorm, max\_pooling, layer denso da 32. Dropout a 0.5

   1. Risultati migliori ma con 30 epoche loss che oscilla
   2. Aggiungo la class\_weigth --> stesso problema loss che oscilla (anche overfitting)
   3. Aggiungo early stopping e aumento epoche--> risultati migliori però oscilla sempre
   4. Aumento la patience dell'early stopping --> sempre problemi con l'oscillazione,
   5. Aumento la patiencce anche del reduce\_lr --> loss che oscilla
   6. riducendo il dropout migliora (oscilla meno), ridotte anche le patience --> provo a rimuovere il dropout
   7. dropout rimosso, più o meno stessi risultati
2. Rimosso il max\_pooling perché sennò zero\_mask non funziona sul layer denso --> decisamente peggio



26/06

I modelli GRU, LSTM e RNN non funzionano (val\_loss troppo alta) --> passo ai transformer. Utilizzo una versione tiny di BERT addestrata sulla lingua italiana
disponibile su HuggingFace

1. Con 30 epoche e lr=0.001 loss sul training = 0.15

Il modello pre-addestrato funziona bene, sono passato ai transformer ma ho ottenuto risultati simili a GRU e LSTM



29/09

Provo ad utilizzare gli embedding già addestrati di FastText. Il problema è dovuto che il dataset è piccolo e quindi i modelli fanno fatica a imparare sia la lingua italiana che successivamente gli esercizi

I vettori di FastText hanno dimensione 300, provo ad utilizzare la PCA per ridurli a 70

1. Provo con LSTM

   1. scarsi risultati, val\_loss sul 1.3
   2. Provo LSTM ma senza PCA --> peggio ancora
   3. Provo ad usare PCA ma aumentare la profondità della rete 64:32 --> val\_loss su 1.2
   4. Provo senza PCA con 64:32
2. Provo GRU

   1. GRU con 48 neuroni e embedding da 70 la migliore fino ad adesso
   2. GRU con 64:64 e embedding a 300 --> peggio



30/09

Tolgo i fasttext, provo a lavorare sul dataset con pre-processamento del testo e riduzione dei token

mask\_zero sembra fondamentale perché ci sono alcuni esercizi che sono molto corti



1/07

Vado con i transformer pre-addestrati. Vado con due modelli di bert, uno tiny l'altro meno

Ottenuto delle loss migliori

Modello tiny:

1. Provo ad diminuire il lr con lr\_with\_plateu --> la loss si aggira sempre su 0.9
2. Cambiato max\_lenght dei token da 70 a 100 --> risultati migliori, provo ad aumentare ancora
3. Se aumento va peggio

Modello bert\_uncased-italian

1. Provo con il setup max\_lenght=100 e lr con plateu --> 0.7 di loss (con questo modello conviene fare poche epoche)
2. Provo a cambiare il weight\_decay a 0.1 --> loss a 0.70 (la migliore)
3. batch\_size da 16 a 32 --> loss di 0.68



3/07

Sto facendo pre-processing sul dataset, normalizzazione punteggiatura e rimozione parola figura

Ottenuto con bert-uncased-xxl e parametri come 3., loss --> 0.66

Provo a fare altro pre-processing, miglioramento della punteggiatura e spazi e non vado a rimuovere la parola figura

