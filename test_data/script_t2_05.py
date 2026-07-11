import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import streamlit as st

# ================= CONFIG =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/flowers"
MODEL_PATH = "flower_resnet18.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
PATIENCE = 5
SEED = 42

CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
NUM_CLASSES = len(CLASSES)

torch.manual_seed(SEED)

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= DATASET WRAPPER =================
class WrappedDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

# ================= DATA LOAD =================
def prepare_data():
    dataset = datasets.ImageFolder(DATA_DIR)

    train_len = int(0.8 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_s, val_s, test_s = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(WrappedDataset(train_s, train_tf), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WrappedDataset(val_s, test_tf), BATCH_SIZE)
    test_loader = DataLoader(WrappedDataset(test_s, test_tf), BATCH_SIZE)

    return train_loader, val_loader, test_loader

# ================= MODEL =================
class FlowerModel(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

# ================= TRAIN =================
def train_model():
    model = FlowerModel().to(DEVICE)
    train_loader, val_loader, _ = prepare_data()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_loss = float("inf")
    patience_counter = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += loss_fn(model(x), y).item()

        val_loss /= len(val_loader)
        st.write(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            st.write("Early stopping triggered!")
            break

    model.load_state_dict(best_weights)
    return model

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = FlowerModel().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        model = train_model()
    model.eval()
    return model

model = load_model()

# ================= PREDICT =================
def predict(image):
    img = test_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)[0]
        top3 = torch.topk(probs, k=3)
    predictions = [(CLASSES[idx], float(prob)) for idx, prob in zip(top3.indices, top3.values)]
    return predictions

# ================= STREAMLIT UI =================
st.set_page_config(page_title="🌸 Flower Classifier", layout="wide")
st.title("🌸 High Accuracy Flower Classification")

uploaded = st.file_uploader("Upload a flower image", ["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        preds = predict(img)
        st.success(f"Top Prediction: **{preds[0][0].upper()}** with **{preds[0][1]*100:.2f}%** confidence")
        st.info("Other Top Predictions:")
        for cls, prob in preds[1:]:
            st.write(f"{cls.upper()} : {prob*100:.2f}%")
