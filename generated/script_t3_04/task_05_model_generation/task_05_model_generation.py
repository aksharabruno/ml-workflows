from dependency import *  # noqa: F401,F403


def model_generation_5(label_names, num_labels, train_loader, val_loader):
    # --- 6. Modèle ---
    model = build_model(num_labels=num_labels)
    model = model.to(device)

    # --- 7. Optimiseur AdamW ---
    # On exclut le bias et les LayerNorm des la régularisation L2 (bonne pratique)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=args.lr)

    # --- 8. Scheduler linéaire avec warmup ---
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"[main] Steps totaux : {total_steps} | Warmup : {warmup_steps}")

    # --- 9. Fonction de perte ---
    # CrossEntropyLoss attend des logits (pas de softmax) et des labels entiers
    criterion = nn.CrossEntropyLoss()

    # --- 10. Dossier de sauvegarde ---
    os.makedirs(args.model_dir, exist_ok=True)
    best_model_path = os.path.join(args.model_dir, "best_model.pt")
    best_val_loss   = float("inf")

    # --- 11. Historique pour les courbes ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_accuracy": [], "val_accuracy": [],
        "val_f1": [],
    }

    # --- 12. Boucle d'entraînement ---
    print(f"\n{'='*60}")
    print(f"  Début de l'entraînement ({args.epochs} epochs)")
    print(f"{'='*60}\n")

    final_preds  = []
    final_labels = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, criterion
        )

        # Validation
        val_loss, val_acc, val_f1, val_preds, val_labels = eval_epoch(
            model, val_loader, device, criterion
        )

        # Mise à jour de l'historique
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Learning rate courant (premier groupe de paramètres)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  train_loss={train_loss:.4f} | train_acc={train_acc:.4f}\n"
            f"  val_loss  ={val_loss:.4f}   | val_acc  ={val_acc:.4f}  | val_f1={val_f1:.4f}\n"
            f"  lr={current_lr:.2e}"
        )

        # Sauvegarde du meilleur modèle (critère : val_loss minimale)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_preds   = val_preds
            final_labels  = val_labels
            save_checkpoint(
                model,
                best_model_path,
                metadata={
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "label_names": label_names,
                    "num_labels": num_labels,
                    "max_length": args.max_length,
                },
            )
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        print()

    return best_model_path, final_labels, final_preds, history
