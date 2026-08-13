#!/usr/bin/env python3
"""
train.py

Parameter-efficient fine-tuning of BERT for binary classification using
4-bit quantization + LoRA (QLoRA-style).

Run from terminal:
  python train.py \
    --model_name bert-base-uncased \
    --dataset_name dipanjanS/imdb_sentiment_finetune_dataset20k \
    --output_dir ./qlora_bert_checkpoint \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --logging_steps 50 \
    --report_to wandb

Adjust batch sizes depending on GPU VRAM.

"""
import argparse
import os
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    #BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Optional: bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    from bitsandbytes.optim import AdamW8bit
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="dipanjanS/imdb_sentiment_finetune_dataset20k")
    parser.add_argument("--output_dir", type=str, default="./qlora_bert_out")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="none")  # set to "wandb" to log
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


args = parse_args()
torch.manual_seed(args.seed)

# ---------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------
print("Loading tokenizer:", args.model_name)
tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)


# ---------------------------------------------------------
# ⚠️ QLoRA warning for BERT (important)
# ---------------------------------------------------------
print(
    "⚠️ NOTE: QLoRA is experimental for encoder-only models like BERT.\n"
    "If you encounter instability, consider fp16 LoRA instead."
)

# -------------------------------
# Load BERT model in 4-bit + prepare for k-bit training
# -------------------------------
from transformers import BitsAndBytesConfig


# Print sample GPU memory (nvidia-smi) and PyTorch summary if GPU available
if torch.cuda.is_available():
    try:
        print("nvidia-smi output (summary):")
        os.system("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv -l 1 -n 1")
    except Exception:
        pass


def compute_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return metric_acc.compute(predictions=preds, references=labels)
