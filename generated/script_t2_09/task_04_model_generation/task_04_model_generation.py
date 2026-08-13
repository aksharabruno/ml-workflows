from dependency import *  # noqa: F401,F403


def model_generation_4(X_train_ids, X_train_mask, X_train_type, y_train):
    # 5. Build Model utilizing TF Hub BERT
    print("Step 5: Building BERT Classifier Model...")

    input_word_ids = keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")
    input_mask = keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_mask")
    input_type_ids = keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_type_ids")

    bert_inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    # Load pretrained small BERT from TF Hub
    bert_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2"
    bert_layer = hub.KerasLayer(bert_url, trainable=True, name="BERT_encoder")
    bert_outputs = bert_layer(bert_inputs)
    pooled_output = bert_outputs["pooled_output"] # representation of CLS token

    x = keras.layers.Dropout(0.3)(pooled_output)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(len(CLASSES), activation='softmax')(x)

    model = keras.Model(
        inputs=[input_word_ids, input_mask, input_type_ids],
        outputs=output,
        name="BERT_Emotion_Classifier"
    )

    # Compile with fine-tuning learning rate (3e-5) for updating BERT weights safely
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 6. Train Model
    print(f"\nStep 6: Fine-tuning BERT model for {EPOCHS} epochs...")
    history = model.fit(
        [X_train_ids, X_train_mask, X_train_type], y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1
    )

    # Save fine-tuned model
    model_path = "emotion_classification_bert.keras"
    model.save(model_path)
    print(f"\n[OK] Model saved to '{model_path}'")

    return history, model
