from dependency import *  # noqa: F401,F403


def model_generation_5(train_loader, val_loader):
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

    return model
