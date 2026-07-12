from dataset import getDataset
from config import TRAIN_DIR, VALIDATION_DIR, EMBEDDING_DIR
import keras
import numpy as np
from embeddings import createEmbeddingDict
from sklearn.decomposition import PCA
from lstm_model import build_lstm_model
from save_metrics import salva_metriche
import tensorflow as tf


dataset_train = getDataset(TRAIN_DIR)
dataset_validation = getDataset(VALIDATION_DIR)

#dataset_train = dataset_train.shuffle(buffer_size=500, reshuffle_each_iteration=True)


token_vocabolary = keras.layers.TextVectorization(
    max_tokens=3000,
    output_mode='int',
    output_sequence_length=100
)

#Estrazione delle parole
dataset_text = dataset_train.map(lambda x,y: x)
token_vocabolary.adapt(dataset_text)

lstm_model = build_lstm_model(token_vocabolary)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-5)

optimizer = keras.optimizers.Adam(learning_rate=0.001)

lstm_model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

num_epochs = 150
history = lstm_model.fit(
    dataset_train,
    epochs = num_epochs,
    validation_data= dataset_validation,
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

val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_accuracy'][-1]
train_loss = history.history['loss'][-1]
train_acc = history.history['accuracy'][-1]

if (response.upper() == 'Y'):
    salva_metriche(history, lstm_model, num_epochs)
