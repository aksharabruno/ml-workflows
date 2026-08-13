from dependency import *  # noqa: F401,F403


def model_evaluation_5(X_test_ids, X_test_mask, X_test_type, history, model, y_test):
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

