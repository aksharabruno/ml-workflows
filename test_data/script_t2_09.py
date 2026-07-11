import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import tensorflow as tf
import tf_keras as keras
import tensorflow_hub as hub
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
CLASSES = ["joy", "anger", "sadness", "surprise"]
MAX_LEN = 32
BATCH_SIZE = 16
EPOCHS = 15

# ------------------------------------------------------------------ #
# Rich Synthetic Dataset Generator
# ------------------------------------------------------------------ #
def generate_synthetic_dataset():
    raw_dataset = {
        "joy": [
            "I am so happy and excited today!",
            "This is the best news I have ever heard!",
            "We won the championship, this is wonderful!",
            "What a beautiful and sunny day!",
            "I feel so grateful and cheerful.",
            "Success brings so much happiness and delight.",
            "I love spending time with my family and friends.",
            "That joke was hilarious, I cannot stop laughing!",
            "I am absolutely thrilled with the results.",
            "I feel highly motivated and full of life today.",
            "It is a pleasure to meet such wonderful people.",
            "We celebrated our victory with joy and dancing.",
            "My heart is full of peace, love and content.",
            "This achievement makes me feel proud and happy.",
            "I am enjoying this beautiful melody.",
            "Today was a perfect day filled with smiles.",
            "I received a lovely gift from my friend.",
            "I am so glad everything worked out so well.",
            "It's a wonderful feeling to help someone in need.",
            "Laughter and happiness filled the entire room.",
            "I feel blessed and very fortunate.",
            "What a pleasant surprise and lovely evening!",
            "We had a fantastic holiday at the beach.",
            "This success exceeds all my expectations!",
            "I feel so warm and cozy inside.",
            "She has a bright and beautiful smile that lifts everyone.",
            "It is a magnificent day to go for a run.",
            "I am so proud of your hard work and achievements.",
            "Everything is going perfectly, I am overjoyed.",
            "Seeing you succeed makes my heart glow with pride."
        ] * 2,
        "anger": [
            "I am absolutely furious about this situation!",
            "Get out of my face, I hate this!",
            "This is completely unfair and makes me mad.",
            "I cannot believe they did this to me, it's annoying.",
            "Stop irritating me, I am so pissed off.",
            "This bad service is extremely frustrating and unacceptable.",
            "I am so angry that I want to scream!",
            "They lied to my face, and I am disgusted.",
            "His rude behavior is completely intolerable.",
            "I am sick and tired of these constant interruptions.",
            "This is a total disaster, and it's all your fault!",
            "He broke his promise, which makes me rage.",
            "I cannot stand his arrogant and selfish attitude.",
            "They completely ignored my request, I am offended.",
            "This delay is wasting my time and making me mad.",
            "Don't speak to me like that, it's highly disrespectful.",
            "I am extremely annoyed by this stupid mistake.",
            "Their negligence caused a massive failure, I am furious.",
            "Stop making excuses, it makes me even angrier.",
            "This is the worst experience of my life, I am raging.",
            "I am fed up with their terrible attitude.",
            "You have crossed the line, and I won't tolerate it.",
            "He ruined my project, I am absolutely livid.",
            "I feel so much resentment toward their actions.",
            "Stop pushing my buttons, I am about to lose my temper."
        ] * 2 + [
            "I am absolutely furious about this situation!",
            "Get out of my face, I hate this!",
            "This is completely unfair and makes me mad.",
            "I cannot believe they did this to me, it's annoying.",
            "Stop irritating me, I am so pissed off."
        ],
        "sadness": [
            "I feel so lonely and depressed lately.",
            "It is heartbreaking to see them leave.",
            "I am crying because of this terrible loss.",
            "Everything feels gloomy and hopeless today.",
            "I am deeply disappointed and sad.",
            "I miss my old friends and the good times we shared.",
            "The tragedy left everyone in deep sorrow and tears.",
            "I feel completely isolated and abandoned.",
            "My heart is heavy with grief and pain.",
            "It is hard to smile when everything is going wrong.",
            "I am feeling down and just want to be alone.",
            "This failure makes me feel worthless and unhappy.",
            "The cold rain matches the sadness in my heart.",
            "She is going through a very painful divorce.",
            "I feel so sorry for their loss, it's devastating.",
            "Life feels empty and directionless right now.",
            "The memories of that day still bring tears to my eyes.",
            "I am mourning the passing of my beloved pet.",
            "It is a dark and lonely night, full of regret.",
            "I feel rejected and unloved by everyone around me.",
            "This constant struggle is making me lose hope.",
            "I feel so blue and tired of everything.",
            "A deep sense of melancholia settled over the room.",
            "He spoke with a voice full of grief and despair.",
            "I am struggling to find any reason to be happy."
        ] * 2 + [
            "I feel so lonely and depressed lately.",
            "It is heartbreaking to see them leave.",
            "I am crying because of this terrible loss.",
            "Everything feels gloomy and hopeless today.",
            "I am deeply disappointed and sad."
        ],
        "surprise": [
            "Oh my god, I cannot believe my eyes!",
            "Wow! That was completely unexpected!",
            "What a shocking and amazing twist!",
            "I am astonished by this sudden event!",
            "This is an absolute shock to me!",
            "I never expected to see you here today!",
            "He suddenly jumped out of the box, startling me!",
            "The magician's trick left the audience amazed.",
            "I was speechless when they announced my name.",
            "What a surprise! I did not see that coming.",
            "She gasped in shock when she opened the letter.",
            "This sudden change of plans caught me off guard.",
            "I am totally stunned by this beautiful gift!",
            "We stared in disbelief at the sudden turn of events.",
            "It was an astonishing revelation that changed everything.",
            "I am absolutely amazed by this incredible performance!",
            "Who would have thought this could happen?",
            "He won the lottery, it was a mind-blowing shock.",
            "The sudden alarm startled everyone in the building.",
            "I am surprised by your sudden change of heart.",
            "What a bizarre and unexpected coincidence!",
            "The box was empty, which was quite a surprise.",
            "I stood frozen in astonishment at the news.",
            "She threw a surprise birthday party for him.",
            "The sudden thunderclaps startled the quiet neighborhood."
        ] * 2 + [
            "Oh my god, I cannot believe my eyes!",
            "Wow! That was completely unexpected!",
            "What a shocking and amazing twist!",
            "I am astonished by this sudden event!",
            "This is an absolute shock to me!"
        ]
    }
    
    texts, labels = [], []
    for class_name in CLASSES:
        c_idx = CLASSES.index(class_name)
        for phrase in raw_dataset[class_name]:
            texts.append(phrase)
            labels.append(c_idx)
            
    return texts, np.array(labels, dtype=np.int32)

# ------------------------------------------------------------------ #
# Main Execution Pipeline
# ------------------------------------------------------------------ #
def main():
    print("====================================================")
    print("Project 60: Emotion Classification from Text (BERT)")
    print("====================================================")

    # 1. Load Bert Tokenizer
    print("Step 1: Loading pre-trained BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 2. Generate Dataset
    print("Step 2: Preparing text database...")
    texts, labels = generate_synthetic_dataset()
    print(f"  Total samples generated: {len(texts)}")
    print(f"  Classes: {CLASSES}\n")

    # 3. Tokenize Dataset
    print("Step 3: Tokenizing text inputs for BERT...")
    encoded = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='np'
    )
    
    X_ids = encoded['input_ids'].astype(np.int32)
    X_mask = encoded['attention_mask'].astype(np.int32)
    X_type = encoded['token_type_ids'].astype(np.int32)
    
    # 4. Stratified Split (80/20 train/test)
    print("Step 4: Creating train/test split...")
    rng = np.random.RandomState(42)
    indices = np.arange(len(texts))
    
    train_idx, test_idx = [], []
    for c in range(len(CLASSES)):
        class_indices = rng.permutation(np.where(labels == c)[0])
        split_pt = int(len(class_indices) * 0.8)
        train_idx.extend(class_indices[:split_pt])
        test_idx.extend(class_indices[split_pt:])
        
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    X_train_ids, X_test_ids = X_ids[train_idx], X_ids[test_idx]
    X_train_mask, X_test_mask = X_mask[train_idx], X_mask[test_idx]
    X_train_type, X_test_type = X_type[train_idx], X_type[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Test samples : {len(test_idx)}\n")

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

    # 7. Evaluate
    test_loss, test_acc = model.evaluate([X_test_ids, X_test_mask, X_test_type], y_test, verbose=0)
    print(f"[OK] Evaluation Test Accuracy: {test_acc*100:.1f}%\n")
    
    # 8. Test predictions on custom unseen inputs
    test_phrases = [
        "I can't believe we won, this is absolutely incredible!",
        "Get out, this is extremely frustrating and unacceptable behaviour.",
        "I feel so lonely and depressed in this empty room.",
        "She suddenly opened the door and screamed in shock!"
    ]
    
    test_encoded = tokenizer(
        test_phrases,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='np'
    )
    predictions = model.predict([
        test_encoded['input_ids'].astype(np.int32),
        test_encoded['attention_mask'].astype(np.int32),
        test_encoded['token_type_ids'].astype(np.int32)
    ], verbose=0)

    # 9. Draw Dashboard
    print("Step 7: Generating evaluation dashboard...")
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)
    
    # Panel 1: Training Acc & Loss Curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history['accuracy'], color='#2ecc71', linewidth=2, label='Train Acc')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], color='#3498db', linewidth=2, label='Val Acc')
    ax1.axhline(test_acc, color='#e74c3c', linestyle='--', label=f'Test Acc: {test_acc*100:.1f}%')
    ax1.set_title("Fine-Tuning Acc convergence", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Panel 2: Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    test_pred_idx = np.argmax(model.predict([X_test_ids, X_test_mask, X_test_type], verbose=0), axis=1)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int32)
    for true, pred in zip(y_test, test_pred_idx):
        cm[true, pred] += 1
        
    im = ax2.imshow(cm, cmap='Purples', interpolation='nearest')
    ax2.set_title("Emotion Confusion Matrix (Test Set)", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2)
    tick_marks = np.arange(len(CLASSES))
    ax2.set_xticks(tick_marks)
    ax2.set_xticklabels(CLASSES, rotation=20, ha='right', fontsize=9)
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(CLASSES, fontsize=9)
    ax2.set_xlabel('Predicted Emotion', fontweight='bold')
    ax2.set_ylabel('True Emotion', fontweight='bold')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            color = "white" if cm[i, j] > np.max(cm)/2 else "black"
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center", color=color, fontweight='bold')

    # Panel 3: Live Predictions Bar Charts
    ax3 = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(CLASSES))
    colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f1c40f']
    
    for i, (phrase, pred) in enumerate(zip(test_phrases, predictions)):
        y_offset = i * 1.5
        ax3.barh(y_pos + y_offset, pred, height=0.35, color=colors, edgecolor='black', alpha=0.85)
        ax3.text(0.01, y_offset + 1.0, f"\"{phrase[:45]}...\"", fontsize=8.5, fontweight='bold', color='#2c3e50')
        pred_label = CLASSES[np.argmax(pred)]
        ax3.text(0.9, y_offset + 0.5, f"Pred: {pred_label.upper()}", fontsize=9, fontweight='bold', color='#16a085')

    ax3.set_yticks(np.arange(len(test_phrases)) * 1.5 + 0.5)
    ax3.set_yticklabels([f"Phrase {i+1}" for i in range(len(test_phrases))], fontsize=9)
    ax3.set_xlim(0, 1.1)
    ax3.set_title("Live Unseen Text Prediction Confidence Profiles", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Softmax Probability")
    
    # Legend for class colors
    patches = [mpatches.Patch(color=colors[i], label=CLASSES[i]) for i in range(len(CLASSES))]
    ax3.legend(handles=patches, loc='lower right', fontsize=8.5)

    # Panel 4: BERT Architecture Pipeline Flowchart
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_title("TF Hub BERT Fine-Tuning Pipeline", fontsize=12, fontweight='bold')

    boxes = [
        (5.0, 9.2, "Raw Text Inputs", "E.g., \"I am so happy and excited today!\"", "#34495e"),
        (5.0, 7.2, "BERT Preprocessor (Hugging Face)", "Generates word ids, attention mask, type ids", "#2980b9"),
        (5.0, 5.2, "Pretrained BERT Layer (TF Hub)", "Encoder (L-2, H-128, A-2) extracts pooled representation", "#8e44ad"),
        (5.0, 3.2, "Dense Classification Head + BN + Dropout", "Maps 128-dim embedding to 32-dim features", "#27ae60"),
        (5.0, 1.2, "Softmax Outputs", "[joy, anger, sadness, surprise] Probabilities", "#d35400")
    ]
    for x, y_coord, title, desc, color in boxes:
        ax4.add_patch(mpatches.FancyBboxPatch(
            (x - 3.8, y_coord - 0.65), 7.6, 1.3,
            boxstyle="round,pad=0.08", facecolor=color, alpha=0.15, edgecolor=color, linewidth=2.0))
        ax4.text(x, y_coord + 0.15, title, ha='center', va='center', fontsize=9.5, color=color, fontweight='bold')
        ax4.text(x, y_coord - 0.35, desc, ha='center', va='center', fontsize=7.5, color='#444444')
        if y_coord > 2.0:
            ax4.annotate('', xy=(x, y_coord - 0.73), xytext=(x, y_coord - 1.48),
                         arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=2.0))

    fig.suptitle("Project 60: Emotion Classification from Text (BERT)\n"
                 f"Pretrained BERT Encoder Fine-Tuning  |  Test Set Accuracy: {test_acc*100:.1f}%",
                 fontsize=14, fontweight='bold', color='#2c3e50')
                 
    output_filename = "emotion_results.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[OK] Evaluation dashboard saved as '{output_filename}'")
    print("====================================================")

if __name__ == "__main__":
    main()
