import keras
import os
import glob
import re


def getDataset(dir):
    
    dataset = keras.utils.text_dataset_from_directory(
        directory = dir,
        batch_size = 32,
        label_mode = 'int',
        shuffle = True
    )

    #dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    return dataset

def showDataset(dataset, token_vocabulary):

    for text_batch, label_batch in dataset.take(1):
        for i in range(2):
            text = text_batch.numpy()[i].decode("utf-8")
            id_categoria = label_batch.numpy()[i]

            print(f"🔹 ESEMPIO {i+1}")
            print(f"Testo: {text}")
            print(f"ID Categoria: {id_categoria}")
            print("-" * 10)
            token_text = token_vocabulary([text_batch[i]])
            print("Testo tokenizzato: ",token_text.numpy())

def showToken(dataset, token_vocabolary):
    for text_batch, _ in token_vocabolary.take(1):
        for i in range(2):
            token_text = token_vocabolary([text_batch[i]])
            text = text_batch.numpy()[i].decode("utf-8")
            print("Testo originale:\n",text)
            print("Testo tokenizzato: ",token_text.numpy())


def getTextAndLabelsFromDataset(dir: str):
    texts, labels = [], []
    sub_folders = os.listdir(dir)

    for label in sub_folders:
        label_dir = os.path.join(dir,label)
        for text in glob.glob(os.path.join(label_dir, "*.txt")):
            with open(text, "r", encoding="utf-8") as f:
              text = f.read()
              #rimozione cifra numero in numero
              text = re.sub(r'\d+([.,]\d+)?', 'numero', text)
              # Normalizza apostrofi e virgolette tipografiche in ASCII
              text = text.replace("’", "'").replace("‘", "'")
              text = text.replace("“", '"').replace("”", '"')
              # Rimuove spazio prima della punteggiatura (artefatto della rimozione di "figura")
              text = re.sub(r'\s+([,.;:!?)])', r'\1', text)
              # Rimuove preposizioni rimaste "orfane" subito prima di punteggiatura
              # (es. "vedete in ," -> "vedete in," -> ancora sporco: meglio rimuovere "in" orfano)
              text = re.sub(r'\b(in|a|di)\s*([,.)])', r'\2', text, flags=re.IGNORECASE)
              # Collassa spazi multipli
              text = re.sub(r'\s+', ' ', text).strip()
            if text:
                texts.append(text)
                labels.append(label)

    return texts, labels