import joblib
import numpy as np
import torch

CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
    'tile', 'toothbrush', 'transistor', 'zipper', 'wood'
]

# 1. Save PyTorch Feature Extractor weights once
torch.save(feature_extractor.state_dict(), 'feature_extractor.pth')

# 2. Extract features and train k-NN for ALL categories
for cat in CATEGORIES:
    print(f"Processing category: {cat}...")
    train_dataset = MvtecDataset(
        root_dir='/kaggle/input/datasets/ipythonx/mvtec-ad', 
        category=cat, 
        is_train=True, 
        transform=data_transform, 
        use_alignment=False
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    train_features_list = []
    with torch.no_grad():
        for images, _ in train_loader:
            images = images.to(device)
            feats = feature_extractor(images)
            B, C, H, W = feats.shape
            feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
            train_features_list.append(feats.cpu().numpy())

    train_features = np.concatenate(train_features_list, axis=0)
    knn_clf = NearestNeighbors(n_neighbors=1, metric='cosine', n_jobs=-1)
    knn_clf.fit(train_features)

    # Save each category's k-NN model
    joblib.dump(knn_clf, f'knn_clf_{cat}.joblib')

print("All categories trained and saved successfully!")
