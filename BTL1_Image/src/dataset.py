import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2 # Dùng v2 cho RandAugment
from PIL import Image
from sklearn.model_selection import train_test_split
import numpy as np

class BloodCellDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, random_seed=42, use_advanced_aug=False):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    
    all_image_paths, all_labels = [], []
    for cls_name in class_names:
        cls_path = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(cls_path, img_name))
                all_labels.append(class_to_idx[cls_name])
                
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_image_paths, all_labels, test_size=val_split, random_state=random_seed, stratify=all_labels 
    )
    
    # --- AUGMENTATION & ROBUSTNESS ---
    if use_advanced_aug:
        train_transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandAugment(num_ops=2, magnitude=9), # Tự động áp dụng nhiễu, biến dạng ảnh
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = BloodCellDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = BloodCellDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Tính Class Weights (Dùng cho hàm Loss nếu không dùng Focal Loss)
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_names)
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    return train_loader, val_loader, class_names, class_weights_tensor