import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import joblib
import cv2
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    # 1. Load ResNet18 backbone directly from torchvision
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(device)
    feature_extractor.eval()
    
    # 2. Download k-NN model from your Hugging Face Space
    knn_path = hf_hub_download(
        repo_id="AnandhuMadhu123/bottle-knn-model", 
        filename="knn_clf_bottle.joblib",
        repo_type="space"
    )
    knn_clf = joblib.load(knn_path)
    
    return feature_extractor, knn_clf

feature_extractor, knn_clf = load_models()

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("MVTec AD - Bottle Anomaly Detector")
uploaded_file = st.file_uploader("Upload a bottle image...", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process image
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats = feature_extractor(img_tensor)
        B, C, H, W = feats.shape
        feats_reshaped = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        
        # Calculate patch distance
        distances, _ = knn_clf.kneighbors(feats_reshaped)
        anomaly_map = distances.reshape(H, W)
        anomaly_score = np.max(anomaly_map)
        
    # Display Results
    st.metric("Max Anomaly Score", f"{anomaly_score:.4f}")
    
    # Render Heatmap
    heatmap = cv2.resize(anomaly_map, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    st.image(heatmap, caption="Anomaly Heatmap", use_column_width=True)
