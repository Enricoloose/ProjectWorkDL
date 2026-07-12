from dataset import *
from config import TRAIN_DIR, VALIDATION_DIR, EMBEDDING_DIR
import keras
import numpy as np
from embeddings import createEmbeddingDict
from sklearn.decomposition import PCA
from lstm_model import build_lstm_model_fasttext
from save_metrics import salva_metriche
import tensorflow as tf


train_text, train_labels_str = getTextAndLabelsFromDataset(TRAIN_DIR)
val_text, val_labels_str = getTextAndLabelsFromDataset(VALIDATION_DIR)

#dataset_train = dataset_train.shuffle(buffer_size=500, reshuffle_each_iteration=True)

categorie_uniche = sorted(list(set(train_labels_str)))
mappa_etichette = {categoria: i for i, categoria in enumerate(categorie_uniche)}

train_labels = np.array([mappa_etichette[label] for label in train_labels_str])
val_labels = np.array([mappa_etichette[label] for label in val_labels_str])

token_vocabolary = keras.layers.TextVectorization(
    max_tokens=3000,
    output_mode='int',
    output_sequence_length=100
)

#Estrazione delle parol
token_vocabolary.adapt(train_text)

vocabolary = token_vocabolary.get_vocabulary()
word_index = dict(zip(vocabolary, range(len(vocabolary))))

#da fasttexr
embedding_dim = 300
embedding_matrix = np.zeros((token_vocabolary._max_tokens, embedding_dim))

embedding_dict = createEmbeddingDict(EMBEDDING_DIR)

for index, word in enumerate(vocabolary):
    if word in embedding_dict:
        embedding_matrix[index] = embedding_dict[word]

#Utilizzo la PCA per ridimensionare gli embedding da 300 a 100
#pca = PCA(n_components=100)
#embedding_matrix = pca.fit_transform(embedding_matrix)

BATCH_SIZE = 16
dataset_train = tf.data.Dataset.from_tensor_slices((train_text, train_labels)).shuffle(len(train_text)).batch(BATCH_SIZE)
dataset_validation = tf.data.Dataset.from_tensor_slices((val_text, val_labels)).batch(BATCH_SIZE)

lstm_model = build_lstm_model_fasttext(embedding_matrix)

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
