from dependency import *  # noqa: F401,F403


def model_generation_11(accuracy, classification_rep, final_svm_model, label_encoder):
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_rep)

    # Save the model and label encoder
    print("Saving the model and label encoder...")
    joblib.dump(final_svm_model, 'best_svm_cyberbullying_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

