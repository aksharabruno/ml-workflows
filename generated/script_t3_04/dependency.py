"""
train.py
--------
Boucle d'entraînement PyTorch manuelle pour le fine-tuning de BERT.
Contient : train_epoch, eval_epoch, et la fonction main() complète.

Usage :
    python train.py --data_path data/bbc-news-data.csv --epochs 3
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import TextClassificationDataset, load_bbc_dataset
from model import build_model, load_tokenizer, save_checkpoint
from utils import set_seed, compute_metrics, plot_learning_curves, plot_confusion_matrix


# ---------------------------------------------------------------------------
# Hyperparamètres par défaut (modifiables via argparse)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    data_path   = "data/bbc-news-data.csv",
    model_dir   = "checkpoints",
    max_length  = 256,   # justification : médiane ~326 mots ≈ 400 tokens → 256 capture ~75% du texte
    batch_size  = 16,    # compromis VRAM / stabilité
    epochs      = 3,     # BERT converge vite ; au-delà → risque d'overfitting
    lr          = 2e-5,  # learning rate typique pour fine-tuning BERT
    weight_decay= 0.01,  # régularisation L2 via AdamW
    warmup_ratio= 0.1,   # 10% des steps en warmup linéaire
    seed        = 42,
    test_size   = 0.2,   # split 80/20 stratifié
)



"""
Orchestre l'entraînement complet :
  1. Fixation de la seed
  2. Chargement du dataset et split stratifié
  3. Construction des DataLoaders
  4. Initialisation du modèle, optimiseur, scheduler, loss
  5. Boucles d'entraînement et de validation
  6. Sauvegarde du meilleur modèle (best val_loss)
  7. Rapport final et visualisations
"""
# --- 1. Reproductibilité ---
set_seed(args.seed)

# --- 2. Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Device : {device}")
print(f"{'='*60}\n")


# --- 4. Tokenizer ---
tokenizer = load_tokenizer()


parser = argparse.ArgumentParser(description="Fine-tuning BERT pour BBC News")
parser.add_argument("--data_path",    type=str,   default=DEFAULTS["data_path"])
parser.add_argument("--model_dir",    type=str,   default=DEFAULTS["model_dir"])
parser.add_argument("--max_length",   type=int,   default=DEFAULTS["max_length"])
parser.add_argument("--batch_size",   type=int,   default=DEFAULTS["batch_size"])
parser.add_argument("--epochs",       type=int,   default=DEFAULTS["epochs"])
parser.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
parser.add_argument("--weight_decay", type=float, default=DEFAULTS["weight_decay"])
parser.add_argument("--warmup_ratio", type=float, default=DEFAULTS["warmup_ratio"])
parser.add_argument("--seed",         type=int,   default=DEFAULTS["seed"])
parser.add_argument("--test_size",    type=float, default=DEFAULTS["test_size"])

args = parser.parse_args()

# Affichage de la configuration
print("\nConfiguration :")
for k, v in vars(args).items():
    print(f"  {k:15s} = {v}")
print()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Exécute une epoch d'entraînement complète.

    Étapes pour chaque batch :
      1. Forward pass : calcul des logits
      2. Calcul de la loss (CrossEntropyLoss)
      3. Backward pass : calcul des gradients
      4. Gradient clipping (évite l'explosion des gradients)
      5. Mise à jour des poids (optimizer.step)
      6. Mise à jour du scheduler de learning rate
      7. Remise à zéro des gradients (optimizer.zero_grad)

    Args:
        model     : BertForSequenceClassification en mode train
        loader    : DataLoader d'entraînement
        optimizer : AdamW
        scheduler : scheduler linéaire avec warmup
        device    : cpu ou cuda
        criterion : CrossEntropyLoss

    Returns:
        avg_loss : loss moyenne sur l'epoch
        accuracy : accuracy sur l'epoch
    """
    model.train()  # active le dropout et BatchNorm

    total_loss  = 0.0
    correct     = 0
    total       = 0

    # tqdm affiche la barre de progression dans le terminal
    progress = tqdm(loader, desc="  [Train]", leave=False)

    for batch in progress:
        # Transfert des données vers le bon device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        # --- Forward pass ---
        # BertForSequenceClassification retourne un objet SequenceClassifierOutput
        # qui contient : loss (si labels fournis), logits, hidden_states, attentions
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits  # shape : (batch_size, num_labels)

        # --- Calcul de la loss ---
        # CrossEntropyLoss attend des logits (non-softmaxés) et des labels entiers
        loss = criterion(logits, labels)

        # --- Backward pass ---
        loss.backward()

        # Gradient clipping : empêche les gradients trop grands (max_norm=1.0)
        # Particulièrement important pour BERT avec son grand nombre de couches
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- Mise à jour ---
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # --- Statistiques ---
        total_loss += loss.item()
        preds       = torch.argmax(logits, dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        # Affichage live dans la barre de progression
        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, float, list, list]:
    """
    Évalue le modèle sur le loader de validation.

    IMPORTANT : on bascule le modèle en mode eval (model.eval()) et on
    désactive le calcul des gradients (torch.no_grad()) pour :
      - Désactiver le dropout (reproductibilité des prédictions)
      - Ne pas construire le graphe de calcul → économie de mémoire

    Args:
        model     : BertForSequenceClassification
        loader    : DataLoader de validation
        device    : cpu ou cuda
        criterion : CrossEntropyLoss

    Returns:
        avg_loss : loss moyenne
        accuracy : accuracy
        f1_macro : F1-score macro
        all_preds  : liste de toutes les prédictions
        all_labels : liste de tous les vrais labels
    """
    model.eval()  # désactive dropout

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():  # pas de calcul de gradients pendant la validation
        progress = tqdm(loader, desc="  [Val]  ", leave=False)
        for batch in progress:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            loss   = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics["accuracy"], metrics["f1_macro"], all_preds, all_labels
