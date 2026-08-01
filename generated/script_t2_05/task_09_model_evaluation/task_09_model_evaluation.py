from dependency import *  # noqa: F401,F403


def model_evaluation_9(img):
    if st.button("Classify"):
        preds = predict(img)
        st.success(f"Top Prediction: **{preds[0][0].upper()}** with **{preds[0][1]*100:.2f}%** confidence")
        st.info("Other Top Predictions:")
        for cls, prob in preds[1:]:
            st.write(f"{cls.upper()} : {prob*100:.2f}%")
