import numpy as np
import tensorflow as tf

#apertura file embedding pre-addestrati
def createEmbeddingDict(dir):
    embedding_dict = {}

    with open(dir, "r", encoding="utf-8") as f:
        #prendo parole e numeri in due variabili diverse
        counter = 0
        for line in f:
            sub = line.split()
            word = sub[0]
            tokens = np.asarray(sub[1:], dtype='float32')

            embedding_dict[word] = tokens

    return embedding_dict

