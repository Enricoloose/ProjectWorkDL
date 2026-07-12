from datetime import datetime
import keras

def salva_metriche(train_loss, train_acc, val_loss, val_acc, rnn_model : keras.Sequential, num_epochs,
                   file_path="metrics.txt"):
    """
    Salva le metriche in un file di testo con timestamp italiano.
    """

    # Timestamp formato italiano: GG/MM/AAAA HH:MM:SS
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Data: {timestamp}\n")
        f.write(f"Epoche: {num_epochs}\n")
        f.write(f"train_loss: {train_loss}\n")
        f.write(f"train_acc: {train_acc}\n")
        f.write(f"val_loss: {val_loss}\n")
        f.write(f"val_acc: {val_acc}\n")
        rnn_model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("-" * 20 + "\n")
