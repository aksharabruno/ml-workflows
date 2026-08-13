from dependency import *  # noqa: F401,F403


def model_evaluation_6(model, test_tf):
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
