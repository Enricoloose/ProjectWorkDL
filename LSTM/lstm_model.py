import keras

#parametri
dropout = 0.5
output_dim = 16
l2_reg = 0.001

def build_lstm_model(vocabulary_layer : keras.layers.TextVectorization):
    return keras.Sequential([

        vocabulary_layer,

        keras.layers.Embedding(input_dim=vocabulary_layer._max_tokens, 
                               output_dim = output_dim,
                               mask_zero= True
                               ),

        keras.layers.LayerNormalization(),

        keras.layers.LSTM(
            32, 
            activation='tanh', 
            dropout=dropout, 
            return_sequences=True, 
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        ),

        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),

        keras.layers.LSTM(
            16,
            activation='tanh', 
            dropout=dropout, 
            return_sequences=False, 
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        ),

        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(units=7, activation='softmax')

    ])

def build_lstm_model_fasttext(embedding_matrix):
    return keras.Sequential([

        keras.layers.Input(shape=(None,), dtype='float32'),

        keras.layers.Embedding(
            input_dim=embedding_matrix.shape[0], 
            output_dim=embedding_matrix.shape[1],
            mask_zero= True,
            weights=[embedding_matrix],
            trainable = False
        ),

        keras.layers.LayerNormalization(),

        keras.layers.LSTM(
            32, 
            activation='tanh', 
            dropout=dropout, 
            return_sequences=False, 
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        ),

        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(
            16,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
        ),

        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),

        keras.layers.Dense(units=7, activation='softmax')
    ])


