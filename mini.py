import streamlit as st
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import requests
import os

# -------------------------------
# Download ONNX model to local
# -------------------------------
onnx_model_url = "https://huggingface.co/alanjoshua2005/spam-sms-india-onnx/resolve/main/bert_sms_detector.onnx"
onnx_model_path = "bert_sms_detector.onnx"

@st.cache_resource
def download_model():
    if not os.path.exists(onnx_model_path):
        with open(onnx_model_path, "wb") as f:
            f.write(requests.get(onnx_model_url).content)
    return onnx_model_path

# -------------------------------
# Load tokenizer + ONNX session
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
st.write("Enter any SMS text below and the model will detect whether it's **Spam** or **Ham**.")

user_input = st.text_area("Enter SMS text:", height=120)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please type a message.")
    else:
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

        outputs = session.run(None, onnx_inputs)
        logits = outputs[0]
        predicted_class = int(np.argmax(logits, axis=1)[0])

        class_map = {0: "Ham (Not Spam)", 1: "Spam"}
        label = class_map[predicted_class]

        if predicted_class == 1:
            st.error(f"🔴 **Prediction: {label}**")
        else:
            st.success(f"🟢 **Prediction: {label}**")
