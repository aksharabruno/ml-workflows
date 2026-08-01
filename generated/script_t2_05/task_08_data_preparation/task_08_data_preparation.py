from dependency import *  # noqa: F401,F403


def data_preparation_8():
    uploaded = st.file_uploader("Upload a flower image", ["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

    return img
