from dataset import AnimalDataset
from model import CNN
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
import os
from config import Config

def train() :

    # Load configuration from config.py
    IMAGE_SIZE = Config.IMAGE_SIZE
    BATCH_SIZE = Config.BATCH_SIZE
    EPOCHS = Config.EPOCHS
    LEARNING_RATE = Config.LEARNING_RATE
    
    DATASET_PATH = Config.DATASET_PATH
    LAST_SAVE_PATH = Config.LAST_SAVE_PATH
    BEST_SAVE_PATH = Config.BEST_SAVE_PATH

    # Create transform pipeline: Resize images to 224x224 and convert to Tensor
    transform = Compose([
        Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ToTensor()
    ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create TRAIN dataset
    dataset_train = AnimalDataset(
        root=DATASET_PATH,
        transform=transform,
        train=True
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # Create VAL dataset
    dataset_val = AnimalDataset(
        root=DATASET_PATH,
        transform=transform,
        train=False
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    NUM_CLASS = len(dataset_train.classes)
    model = CNN(num_classes = NUM_CLASS)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    epochs = EPOCHS
    start_epoch = 0
    
    # === AUTO RESUME TRAINING ===
    if os.path.exists(LAST_SAVE_PATH):
        print(f"Found {LAST_SAVE_PATH}! Loading checkpoint to resume training...")
        checkpoint = torch.load(LAST_SAVE_PATH, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []})
        print(f"Successfully loaded! Resuming training from Epoch {start_epoch + 1}...")
    else:
        print("No previous checkpoint found. Starting training from scratch (Epoch 1)...")
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        
    for epoch in range(start_epoch, epochs):

        # train
        model.train() 
        
        train_all_labels = []
        train_all_predictions = []
        train_all_losses = []
        
        pbar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}", colour = "green")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            # --- Track predictions for Training Accuracy ---
            prediction = torch.argmax(outputs, dim=1).tolist()
            train_all_predictions.extend(prediction)
            train_all_labels.extend(labels.tolist())
            train_all_losses.append(loss.item())
            
            pbar.set_description(f"Epoch {epoch+1}/{epochs}. Loss {loss.item():.4f}")
            
        train_acc = accuracy_score(train_all_labels, train_all_predictions)
        avg_train_loss = np.mean(train_all_losses)
        
        # validation
        model.eval() 
        
        all_labels = []
        all_predictions = []
        all_losses = []

        val_pbar = tqdm(dataloader_val, desc=f"Validation {epoch+1}", colour="cyan")
        
        with torch.no_grad(): 
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                prediction = torch.argmax(outputs, dim=1).tolist()

                all_predictions.extend(prediction)
                all_labels.extend(labels.tolist())
                all_losses.append(loss.item())
                
                val_pbar.set_description(f"Validation {epoch+1}. Loss {loss.item():.4f}")
                
        # Calculate Validation Accuracy
        acc = accuracy_score(all_labels, all_predictions)
        avg_val_loss = np.mean(all_losses)
                
        # Print Epoch summary
        print(f"Epoch [{epoch+1}/{epochs}]: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | Train Acc = {train_acc:.2f}% | Val Acc = {acc:.2f}%")
        
        # SAVE HISTORY FOR PLOTTING
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(acc)
        
        # AUTO SAVE BEST MODEL AFTER EACH EPOCH
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "history": history
        }
        torch.save(checkpoint, LAST_SAVE_PATH)
        
        if acc > best_acc:
            best_acc = acc
            checkpoint["best_acc"] = best_acc
            torch.save(checkpoint, BEST_SAVE_PATH)
            print(f"New Record! Saved best model (Accuracy: {acc:.2f}%) to {BEST_SAVE_PATH}")
            
if __name__ == '__main__' :
    train()
