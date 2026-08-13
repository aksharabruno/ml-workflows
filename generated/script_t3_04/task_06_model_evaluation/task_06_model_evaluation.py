from dependency import *  # noqa: F401,F403


def model_evaluation_6(best_model_path, final_labels, final_preds, history, label_names):
    # --- 13. Rapport final ---
    print(f"\n{'='*60}")
    print("  Rapport final (meilleur modèle)")
    print(f"{'='*60}")
    metrics = compute_metrics(final_labels, final_preds, label_names)
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 macro : {metrics['f1_macro']:.4f}")
    print("\n" + metrics["report"])

    # --- 14. Visualisations ---
    plot_learning_curves(history, save_path=os.path.join(args.model_dir, "learning_curves.png"))
    plot_confusion_matrix(
        final_labels, final_preds, label_names,
        save_path=os.path.join(args.model_dir, "confusion_matrix.png")
    )

    print(f"\n[main] Entraînement terminé. Modèle → {best_model_path}")


