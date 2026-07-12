import keras

#parametri
dropout = 0.5
output_dim = 32
l2_reg = 0.0001

def build_gru_model(vocabulary_layer : keras.layers.TextVectorization):
    return keras.Sequential([

        vocabulary_layer,

        keras.layers.Embedding(
            input_dim=vocabulary_layer._max_tokens, 
            output_dim=output_dim,
            mask_zero= True,
        ),
  
        keras.layers.Bidirectional(
            keras.layers.GRU(64, 
                             activation='tanh', 
                             dropout=dropout, 
                             return_sequences=False, 
                             kernel_regularizer=keras.regularizers.l2(l2_reg))
        ),
        keras.layers.LayerNormalization(),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(32,activation='relu',kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=7, activation='softmax')
    ])

def build_gru_model_fasttext(embedding_matrix, vectorize_layer):
    return keras.Sequential([
        keras.layers.Input(shape=(1,), dtype='string'),

        vectorize_layer,

        keras.layers.Embedding(
            input_dim=embedding_matrix.shape[0], 
            output_dim=embedding_matrix.shape[1],
            mask_zero= True,
            weights=[embedding_matrix],
            trainable = False
        ),
  
        keras.layers.Bidirectional(
            keras.layers.GRU(64,activation='tanh', dropout=dropout, return_sequences=False, kernel_regularizer=keras.regularizers.l2(l2_reg))
            ),
        keras.layers.LayerNormalization(),

        keras.layers.Dropout(dropout),
        keras.layers.Dense(64,activation='relu',kernel_regularizer=keras.regularizers.l2(l2_reg)),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(units=7, activation='softmax')
    ])



