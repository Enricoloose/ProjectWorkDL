from dataset import *
from config import TRAIN_DIR, VALIDATION_DIR
import keras
from preproceesing import text_preproc
from gru_model import build_gru_model
from save_metrics import salva_metriche
import numpy as np

train_dataset = getDataset(TRAIN_DIR)
#train = getDataset(TRAIN_DIR)
validation_dataset = getDataset(VALIDATION_DIR)


#tokenizzazione
token_vocabolary = keras.layers.TextVectorization(
    max_tokens=3000,
    output_sequence_length= 100,
    output_mode= 'int'
)

#crezione vocabolario di token
dataset_text = train_dataset.map(lambda x, y: x)
token_vocabolary.adapt(dataset_text)

#build del modello
gru_model = build_gru_model(token_vocabolary)

optmizer = keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.1)

gru_model.compile(
    optimizer=optmizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=7,min_lr=1e-5)


num_epochs = 150
history = gru_model.fit(
    train_dataset,
    epochs = num_epochs,
    validation_data= validation_dataset,
    callbacks=[early_stopping, reduce_lr],
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

if (response.upper() == 'Y'):
    salva_metriche(history, gru_model, num_epochs)


