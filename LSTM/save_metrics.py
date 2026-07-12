from datetime import datetime
import keras

def salva_metriche(history, rnn_model : keras.Sequential, num_epochs,
                   file_path="metrics.txt"):
    """
    Salva le metriche in un file di testo con timestamp italiano.
    """

    # Timestamp formato italiano: GG/MM/AAAA HH:MM:SS
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy'][-1]
    train_loss = history.history['loss'][-1]
    train_acc = history.history['accuracy'][-1]
    min_val_loss = min(history.history['val_loss'])

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Data: {timestamp}\n")
        f.write(f"{val_loss}\n")
        
        f.write("-" * 20 + "\n")
