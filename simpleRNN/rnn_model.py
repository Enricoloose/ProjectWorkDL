import keras

#parametri
dropout = 0.5
output_dim = 16

def build_rnn_model(vocabulary_layer : keras.layers.TextVectorization):
    return keras.Sequential([
        vocabulary_layer,
        # creazione layer di embedding
        # input vettore di dimensione 3000 [token], output vettore denso di 16
        keras.layers.Embedding(input_dim=vocabulary_layer._max_tokens, output_dim=output_dim),
             
        keras.layers.SimpleRNN(32, return_sequences=True, dropout = dropout),
        keras.layers.SimpleRNN(16, dropout = dropout),

        keras.layers.Dense(16, activation = 'relu'),

        keras.layers.Dense(units=7, activation='softmax')
    ])