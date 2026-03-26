import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore") # Tắt các cảnh báo lặt vặt

from dataset import get_dataloaders
from models import build_model

def calculate_ece(confidences, predictions, labels, num_bins=10):
    """Tính toán Expected Calibration Error (ECE) - Đánh giá độ tin cậy"""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

def unnormalize(tensor):
    """Giải chuẩn hóa ảnh để vẽ hình"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def evaluate_model(model, model_name, dataloader, class_names, device):
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    misclassified_images, misclassified_trues, misclassified_preds = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

            wrong_idx = (preds != labels).nonzero(as_tuple=True)[0]
            for idx in wrong_idx:
                if len(misclassified_images) < 8:
                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_trues.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())

    # --- TÍNH TOÁN 5 METRICS ---
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    ece = calculate_ece(np.array(all_confs), np.array(all_preds), np.array(all_labels))

    # --- LƯU CONFUSION MATRIX ---
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'CM - {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'../docs/cm_{model_name}.png', dpi=200)
    plt.close()

    # --- LƯU ERROR ANALYSIS ---
    if misclassified_images:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, img_tensor in enumerate(misclassified_images):
            img = unnormalize(img_tensor).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[i].imshow(img)
            true_label, pred_label = class_names[misclassified_trues[i]], class_names[misclassified_preds[i]]
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
            axes[i].axis('off')
        plt.suptitle(f'Error Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../docs/errors_{model_name}.png', dpi=200)
        plt.close()

    return acc, f1, precision, recall, ece

if __name__ == '__main__':
    DATA_DIR = '../dataset/bloodcells_dataset'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs('../docs', exist_ok=True)
    
    print(f"BẮT ĐẦU ĐÁNH GIÁ TRÊN THIẾT BỊ: {device}")
    _, val_loader, class_names, _ = get_dataloaders(DATA_DIR, batch_size=32)

    # Khai báo lại 10 mô hình để load weights
    experiments = [
        {"name": "01_ResNet_Freeze", "model": "resnet18", "freeze": True},
        {"name": "02_ResNet_Full", "model": "resnet18", "freeze": False},
        {"name": "03_ResNet_LayerWise", "model": "resnet18", "freeze": False},
        {"name": "04_ViT_Freeze", "model": "vit_b_16", "freeze": True},
        {"name": "05_ViT_LayerWise", "model": "vit_b_16", "freeze": False},
        {"name": "06_MobileNet_LayerWise", "model": "mobilenet_v3", "freeze": False},
        {"name": "07_ResNet_Augment", "model": "resnet18", "freeze": False},
        {"name": "08_ResNet_FocalLoss", "model": "resnet18", "freeze": False},
        {"name": "09_ResNet_Ultimate", "model": "resnet18", "freeze": False},
        {"name": "10_ViT_Ultimate", "model": "vit_b_16", "freeze": False},
    ]

    results = []

    for exp in experiments:
        weight_path = f"../weights/best_{exp['name']}.pth"
        if os.path.exists(weight_path):
            print(f"Đang đánh giá: {exp['name']}...")
            model = build_model(exp['model'], freeze_backbone=exp['freeze'])
            model.load_state_dict(torch.load(weight_path, map_location=device))
            model = model.to(device)
            
            acc, f1, prec, rec, ece = evaluate_model(model, exp['name'], val_loader, class_names, device)
            results.append((exp['name'], acc, f1, prec, rec, ece))
        else:
            print(f"[BỎ QUA] Không tìm thấy {weight_path}")

    # ==========================================
    # IN BẢNG BÁO CÁO TỔNG SẮP
    # ==========================================
    print("\n" + "="*105)
    print(f"{'Tên Mô hình (Thử nghiệm)':<28} | {'Accuracy':<8} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8} | {'ECE':<8}")
    print("-" * 105)
    for name, acc, f1, prec, rec, ece in results:
        print(f"{name:<28} | {acc:.4f}   | {f1:.4f}   | {prec:.4f}    | {rec:.4f}   | {ece:.4f}")
    print("="*105)
    print("=> TẤT CẢ ẢNH CONFUSION MATRIX VÀ ERROR ANALYSIS ĐÃ ĐƯỢC LƯU TRONG THƯ MỤC 'docs/'")