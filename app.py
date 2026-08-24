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

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
    'tile', 'toothbrush', 'transistor', 'zipper', 'wood'
]

# Set defect thresholds per category (Adjust these based on validation testing)
THRESHOLD = 0.38 

@st.cache_resource
def load_feature_extractor():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    feature_extractor = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3
    ).to(device)
    feature_extractor.eval()
    return feature_extractor

@st.cache_resource
def load_knn_model(category):
    knn_path = hf_hub_download(
        repo_id="AnandhuMadhu123/bottle-knn-model", 
        filename=f"knn_clf_{category}.joblib",
        repo_type="space"
    )
    return joblib.load(knn_path)

feature_extractor = load_feature_extractor()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("MVTec AD - Multi-Item Anomaly Detector")

selected_category = st.selectbox("Select Object Category:", CATEGORIES)
uploaded_file = st.file_uploader(f"Upload a {selected_category} image...", type=["jpg", "png", "bmp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width='stretch')
    
    knn_clf = load_knn_model(selected_category)
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        feats = feature_extractor(img_tensor)
        B, C, H, W = feats.shape
        feats_reshaped = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        
        distances, _ = knn_clf.kneighbors(feats_reshaped)
        anomaly_map = distances.reshape(H, W)
        anomaly_score = np.max(anomaly_map)
        
    st.metric("Max Anomaly Score", f"{anomaly_score:.4f}")
    
    # --- Clear Defect Verdict Banner ---
    if anomaly_score > THRESHOLD:
        st.error(f"❌ **STATUS: DEFECTIVE** (Score: {anomaly_score:.4f} > Threshold: {THRESHOLD})")
    else:
        st.success(f"✅ **STATUS: GOOD / NORMAL** (Score: {anomaly_score:.4f} ≤ Threshold: {THRESHOLD})")
    
    # Render Heatmap
    heatmap = cv2.resize(anomaly_map, (224, 224))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    st.image(heatmap, caption="Anomaly Heatmap", width='stretch')
