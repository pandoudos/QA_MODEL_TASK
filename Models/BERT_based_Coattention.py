def create_BERT_coattention_model(pretrained_path, preprocessor_path):
  
  context_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
  question_input = tf.keras.layers.Input(shape=(), dtype=tf.string)

  preprocessor = tf_hub.KerasLayer(preprocessor_path)

  BERT_context_inputs = preprocessor(context_input)
  BERT_question_inputs = preprocessor(question_input)

  BERT = tf_hub.KerasLayer(pretrained_path, trainable=False)

  context_BERT = BERT(BERT_context_inputs)
  context_BERT = context_BERT['sequence_output']
  question_BERT = BERT(BERT_question_inputs)
  question_BERT = question_BERT['sequence_output']

  context_encoder = Bidirectional(LSTM(128, return_sequences=True, dropout=0.33))(context_BERT)

  question_encoder = Bidirectional(LSTM(128, return_sequences=True, dropout=0.33))(question_BERT)

  #Defining Affine matrix L 
  L = tf.linalg.matmul(context_encoder, question_encoder, transpose_a = True)

  question_attention_factors = Activation('softmax')(L)
  context_attention_factors = Activation('softmax')(tf.transpose(L, [0, 2, 1]))

  coattention_context = Attention()([context_encoder, question_attention_factors])
  coattention_question = Attention()([question_encoder, context_attention_factors])

  decoder_inputs = Concatenate(axis=2)([coattention_context, coattention_question])

  decoder_outputs = LSTM(256, return_sequences=False, dropout=0.33)(decoder_inputs)

  coattention_context_pooled = tf.keras.layers.GlobalAveragePooling1D()(coattention_context)

  coattention_question_pooled = tf.keras.layers.GlobalAveragePooling1D()(coattention_question)

  combined_context = Concatenate(axis = 1)([coattention_context_pooled, decoder_outputs])

  output = Dense(5002, activation='softmax')(combined_context)

  model = Model(inputs=[context_input, question_input], outputs=output)

  model.compile(optimizer= tf.keras.optimizers.experimental.AdamW(learning_rate=1e-3), 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
  
  return model