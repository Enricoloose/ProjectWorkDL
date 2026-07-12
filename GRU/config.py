import os

def checkDir(directory):
    if not(os.path.exists(directory)):
        print("Directory non trovata: ",directory)

#PATH per il dataset

CURRENT_DIR = os.getcwd()

PARENT_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(PARENT_DIR,'data','esercizi')
checkDir(DATA_DIR)

TRAIN_DIR = os.path.join(DATA_DIR,'train')
checkDir(TRAIN_DIR)

VALIDATION_DIR = os.path.join(DATA_DIR,'validation')
checkDir(VALIDATION_DIR)

EMBEDDING_DIR = os.path.join(PARENT_DIR,'cc.it.300.vec')
