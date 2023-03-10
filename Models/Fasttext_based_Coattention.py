import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Concatenate, Dot, Activation, RepeatVector, Permute, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_fasttext_coattention_model(embedding_matrix):

  tf.random.set_seed(12)

  context_input = tf.keras.layers.Input(shape=(None,))
  question_input = tf.keras.layers.Input(shape=(None,))

  embedding_layer = tf.keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                              output_dim=embedding_matrix.shape[1],
                                              weights=[embedding_matrix],
                                              trainable=False)

  context_embedded = embedding_layer(context_input)
  question_embedded = embedding_layer(question_input)

  context_encoder = Bidirectional(LSTM(128, return_sequences=True, dropout=0.33))(context_embedded)

  question_encoder = Bidirectional(LSTM(128, return_sequences=True, dropout=0.33))(question_embedded)

  #Defining Affine matrix L 
  L = tf.linalg.matmul(context_encoder, question_encoder, transpose_a = True)

  question_attention_factors = Activation('softmax')(L)
  context_attention_factors = Activation('softmax')(tf.transpose(L, [0, 2, 1]))

  coattention_context = Attention()([context_encoder, question_attention_factors])
  coattention_question = Attention()([question_encoder, context_attention_factors])

  question_summary_representation = tf.linalg.matmul(coattention_question, context_attention_factors) #???

  decoder_inputs = Concatenate(axis=2)([coattention_context, question_summary_representation])

  decoder_outputs = LSTM(256, return_sequences=False, dropout=0.33)(decoder_inputs)

  coattention_context_pooled = tf.keras.layers.GlobalAveragePooling1D()(coattention_context)

  combined_context = Concatenate(axis = 1)([coattention_context_pooled, decoder_outputs])

  output = Dense(5001, activation='softmax')(combined_context)

  model = Model(inputs=[context_input, question_input], outputs=output)

  model.compile(optimizer= tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3), 
              loss = tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
  
  return model

