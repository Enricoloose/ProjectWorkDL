from dataset import getDataset, getTextAndLabelsFromDataset
from config import TRAIN_DIR, VALIDATION_DIR, EMBEDDING_DIR
import keras
import numpy as np
from embeddings import createEmbeddingDict
from sklearn.decomposition import PCA
from gru_model import build_gru_model_fasttext
from save_metrics import salva_metriche
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


dataset_train = getDataset(TRAIN_DIR)
dataset_validation = getDataset(VALIDATION_DIR)

train_text, train_labels_str = getTextAndLabelsFromDataset(TRAIN_DIR)
val_text, val_labels_str = getTextAndLabelsFromDataset(VALIDATION_DIR)

categorie_uniche = sorted(list(set(train_labels_str)))
mappa_etichette = {categoria: i for i, categoria in enumerate(categorie_uniche)}

train_labels = np.array([mappa_etichette[label] for label in train_labels_str])
val_labels = np.array([mappa_etichette[label] for label in val_labels_str])


#tokenizzazione
token_vocabolary = keras.layers.TextVectorization(
    max_tokens=3000,
    output_mode='int',
    output_sequence_length=100
)

#dataset_text = dataset_train.map(lambda x, y: x)

#Creazione vocabolario token
token_vocabolary.adapt(train_text)

def vectorize_text(text, label):
    # tf.expand_dims è spesso necessario per adattare le dimensioni
    import tensorflow as tf
    text = tf.expand_dims(text, -1)
    return token_vocabolary(text), label

vectorized_train_ds = dataset_train.map(vectorize_text)
vectorized_val_ds = dataset_validation.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
vectorized_train_ds = vectorized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
vectorized_val_ds = vectorized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)

vocabolary = token_vocabolary.get_vocabulary()


#da fasttexr
embedding_dim = 300
embedding_matrix = np.zeros((token_vocabolary._max_tokens, embedding_dim))

embedding_dict = createEmbeddingDict(EMBEDDING_DIR)

for index, word in enumerate(vocabolary):
    if word in embedding_dict:
        embedding_matrix[index] = embedding_dict[word]

#Utilizzo la PCA per ridimensionare gli embedding da 300 a 70
pca = PCA(n_components=100)
embedding_matrix = pca.fit_transform(embedding_matrix)

#build model
gru_model = build_gru_model_fasttext(embedding_matrix)



gru_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=10,min_lr=1e-5)


num_epochs = 200
history = gru_model.fit(
    vectorized_train_ds,
    epochs = num_epochs,
    validation_data= vectorized_val_ds,
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



