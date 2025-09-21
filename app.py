
import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json

from model import load_classes,load_model,predict


class_names = load_classes("classes.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("resnet_18.pth", num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load("resnet_18.pth", map_location=device))
model.eval()

st.title("🌾 Rice Variety Classifier with Agronomic Info")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    pred_class, confidence = predict(model, image, class_names, threshold=0.5)
    if pred_class is None:
     st.warning(f"⚠️ Could not confidently identify the rice variety (Confidence: {confidence*100:.2f}%).\n\nTry using a clearer or closer microscopic image or\n upload another image")
    else:
     st.success(f"✅ Predicted Variety: **{pred_class}** ({confidence*100:.2f}%)")

    #pred_class = predict(model, image, class_names)
    #st.success(f"✅ Predicted Variety: **{pred_class}**")

