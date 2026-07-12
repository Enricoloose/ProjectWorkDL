import keras
from dataset import getTextAndLabelsFromDataset
from config import TEST_DIR
import numpy as np

gru_model = keras.models.load_model("model.h5")

test_text, test_labels_str = getTextAndLabelsFromDataset(TEST_DIR)

categorie_uniche = sorted(list(set(test_labels_str)))
mappa_etichette = {categoria: i for i, categoria in enumerate(categorie_uniche)}

test_labels = np.array([mappa_etichette[label] for label in test_labels_str])

print("\nAvvio valutazione sul Test Set...")
loss, acc = gru_model.evaluate(test_text, test_labels, verbose=1)

print(f"\nRisultati Finali sul Test:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")