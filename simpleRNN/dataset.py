import os
import keras


def getDataset(dir):
    
    dataset = keras.utils.text_dataset_from_directory(
        directory = dir,
        batch_size = 32,
        label_mode = 'int',
        shuffle = True
    )

    return dataset

def showDataset(dataset):

    for text_batch, label_batch in dataset.take(1):
        for i in range(2):
            text = text_batch.numpy()[i].decode("utf-8")
            id_categoria = label_batch.numpy()[i]

            print(f"🔹 ESEMPIO {i+1}")
            print(f"Testo: {text}")
            print(f"ID Categoria: {id_categoria}")
            print("-" * 10)

def showToken(dataset, token_vocabolary):
    for text_batch, label_batch in dataset.take(1):
        for i in range(2):
            token_text = token_vocabolary([text_batch[i]])
            text = text_batch.numpy()[i].decode("utf-8")

            print("Testo originale:\n",text)
            print("Testo tokenizzato: ",token_text.numpy())