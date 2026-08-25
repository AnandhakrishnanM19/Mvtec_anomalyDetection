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

THRESHOLDS = {
    'bottle': 0.35,
    'cable': 0.42,
    'capsule': 0.42,
    'carpet': 0.45,
    'grid': 0.40,
    'hazelnut': 0.38,
    'leather': 0.42,
    'metal_nut': 0.41,
    'pill': 0.40,
    'screw': 0.41,    
    'tile': 0.39,
    'toothbrush': 0.35,
    'transistor': 0.40,
    'zipper': 0.43,
    'wood': 0.44
}

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

    threshold = THRESHOLDS.get(selected_category, 0.38) if 'THRESHOLDS' in globals() else 0.38
    
    if anomaly_score > threshold:
        st.error(f"❌ **STATUS: DEFECTIVE** (Score: {anomaly_score:.4f} > Threshold: {threshold})")
    else:
        st.success(f"✅ **STATUS: GOOD / NORMAL** (Score: {anomaly_score:.4f} ≤ Threshold: {threshold})")
    
    heatmap_resized = cv2.resize(anomaly_map, (image.size[0], image.size[1]))
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_norm), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend heatmap with original image
    img_np = np.array(image)
    overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    
    st.image(overlay, caption="Anomaly Heatmap Overlay", width='stretch')
