from config import TRAIN_DIR,VALIDATION_DIR
from dataset import getDataset, showDataset, showToken
import numpy as np
import keras
from rnn_model import build_rnn_model
from save_metrics import salva_metriche


dataset_train = getDataset(TRAIN_DIR)
dataset_validation = getDataset(VALIDATION_DIR)


# Tokenizzazione vocabolario
# parametri considerati in base alla media di parole per esercizio


token_vocabolary = keras.layers.TextVectorization(
    max_tokens=3000,
    output_mode='int',
    output_sequence_length=70
)

dataset_text = dataset_train.map(lambda x, y: x)

# Creazione vocabolario token
token_vocabolary.adapt(dataset_text)

# Build model after vocabulary is created
rnn_model = build_rnn_model(token_vocabolary)

# loss utilizzo la sparse_categorical_cross entropy perché mappa automaticamente le categorie in: one-hot --> valore intero
rnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

num_epochs = 100
history = rnn_model.fit(
    dataset_train,
    epochs = num_epochs,
    validation_data= dataset_validation,
    verbose = 2
    )


#Print delle metriche
mean_loss = sum(history.history['loss'])/num_epochs
mean_acc = sum(history.history['accuracy'])/num_epochs

#print the mean loss and mean accuracy in :.4f format
print(f"Mean loss: {mean_loss:.4f}")
print(f"Mean accuracy: {mean_acc:.4f}")

#print mean val_loss e mean val_accuracy
mean_val_loss = sum(history.history['val_loss'])/num_epochs
mean_val_acc = sum(history.history['val_accuracy'])/num_epochs


print(f"Mean val_loss: {mean_val_loss:.4f}")
print(f"Mean val_accuracy: {mean_val_acc:.4f}")


response = input("Salvare le metriche?")

val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
train_acc = history.history['accuracy'][-1]

if (response.upper() == 'Y'):
    salva_metriche(train_loss, train_acc, val_loss, val_acc, rnn_model, num_epochs)