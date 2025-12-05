import streamlit as st
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import requests
import os

# -------------------------------
# HF model + dataset links
# -------------------------------
MODEL_REPO_URL = "https://huggingface.co/alanjoshua2005/spam-sms-india-onnx"
DATASET_REPO_URL = "https://huggingface.co/datasets/alanjoshua2005/india-spam-sms"

# -------------------------------
# Download ONNX model to local
# -------------------------------
onnx_model_url = f"{MODEL_REPO_URL}/resolve/main/bert_sms_detector.onnx"
onnx_model_path = "bert_sms_detector.onnx"

@st.cache_resource
def download_model():
    if not os.path.exists(onnx_model_path):
        with open(onnx_model_path, "wb") as f:
            f.write(requests.get(onnx_model_url).content)
    return onnx_model_path

# -------------------------------
# Load tokenizer + ONNX runtime
# -------------------------------
@st.cache_resource
def load_resources():
    model_path = download_model()
    tokenizer = AutoTokenizer.from_pretrained("alanjoshua2005/Bert-sms-spam-detector-onnx")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return tokenizer, session

tokenizer, session = load_resources()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📩 SMS Spam Detector (ONNX)")
st.write("This app uses a BERT model fine-tuned on an **Indian SMS Spam** dataset.")

# Display model + dataset links
st.markdown("### Resources")
st.markdown(f"**Model Repository:** [spam-sms-india-onnx]({MODEL_REPO_URL})")
st.markdown(f"**Dataset Repository:** [india-spam-sms]({DATASET_REPO_URL})")
st.write("---")

# Input box
user_input = st.text_area("Enter SMS text:", height=120)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please type a message.")
    else:
        # Tokenize
        inputs = tokenizer(
            user_input,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=64
        )

        onnx_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }

        # Run ONNX inference
        outputs = session.run(None, onnx_inputs)
        logits = outputs[0]
        predicted_class = int(np.argmax(logits, axis=1)[0])

        class_map = {0: "Ham (Not Spam)", 1: "Spam"}
        label = class_map[predicted_class]

        if predicted_class == 1:
            st.error(f"🔴 **Prediction: {label}**")
        else:
            st.success(f"🟢 **Prediction: {label}**")
