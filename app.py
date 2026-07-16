import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = 224
CLASS_NAMES = ["notumor", "pituitary", "glioma", "meningioma"]     
MODEL_PATH = "best_model_eff.keras"

st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write(
    "Upload a brain MRI scan and the model will predict whether it shows "
    "**no tumor**, **pituitary tumor**, **glioma**, or **meningioma**."
)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, convert to array, scale to [0, 1], and add batch dimension."""
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr

model = load_model(MODEL_PATH)
if model is None:
    st.error(
        f"Could not find model file `{MODEL_PATH}` in the app directory."
    )
    st.stop()

st.sidebar.success("✅ EfficientNetV2B0 model loaded")

uploaded_file = st.file_uploader(
    "Upload an MRI image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)
    if model is None:
        st.stop()
    with st.spinner("Running inference..."):
        input_arr = preprocess_image(image)
        preds = model.predict(input_arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_label = CLASS_NAMES[top_idx]
    top_conf = float(preds[top_idx])
    with col2:
        st.subheader("Prediction")
        st.metric(label="Predicted class", value=top_label, delta=f"{top_conf:.1%} confidence")

    st.subheader("Class probabilities")
    sorted_pairs = sorted(zip(CLASS_NAMES, preds), key=lambda x: x[1], reverse=True)
    chart_labels = [p[0] for p in sorted_pairs]
    chart_values = [float(p[1]) for p in sorted_pairs]

    import pandas as pd
    df = pd.DataFrame({"class": chart_labels, "probability": chart_values}).set_index("class")
    st.bar_chart(df)

    with st.expander("Raw probabilities"):
        for label, prob in sorted_pairs:
            st.write(f"**{label}**: {prob:.4f}")
else:
    st.info("👆 Upload a `.jpg` or `.png` MRI image to get a prediction.")
