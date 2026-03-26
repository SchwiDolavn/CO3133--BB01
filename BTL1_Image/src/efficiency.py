import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

from models import build_model
from dataset import get_dataloaders

def count_parameters(model):
    """Đếm tổng số tham số có thể huấn luyện của mô hình (đơn vị: Triệu)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def measure_inference_time(model, device, num_samples=100):
    """
    Đo thời gian suy luận (inference time) trên 1 batch nhỏ.
    Cần chạy khởi động (warm-up) trước để kết quả chính xác.
    """
    model.eval()
    # Giả lập 1 bức ảnh đầu vào chuẩn
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warm-up (Khởi động GPU/CPU)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
            
    # Bắt đầu bấm giờ
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_samples):
            _ = model(dummy_input)
            
    end_time = time.time()
    
    # Tính thời gian trung bình cho 1 ảnh (mili-giây)
    avg_time_ms = ((end_time - start_time) / num_samples) * 1000
    return avg_time_ms

def get_accuracy(model, dataloader, device):
    """Tính nhanh điểm Accuracy trên tập Val"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

if __name__ == '__main__':
    DATA_DIR = '../dataset/bloodcells_dataset'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Đang đo lường trên thiết bị: {device}\n")

    # Chỉ lấy val_loader để tính điểm
    _, val_loader, _, _ = get_dataloaders(DATA_DIR, batch_size=32)

    # Khai báo 3 đại diện để so sánh hiệu quả (Trích từ 10 Test bạn vừa chạy)
    models_to_test = {
        'ResNet18 (Test 03)': {'type': 'resnet18', 'weights': '../weights/best_03_ResNet_LayerWise.pth'},
        'ViT-B/16 (Test 05)': {'type': 'vit_b_16', 'weights': '../weights/best_05_ViT_LayerWise.pth'},
        'MobileNetV3 (Test 06)': {'type': 'mobilenet_v3', 'weights': '../weights/best_06_MobileNet_LayerWise.pth'}
    }

    results = []

    for name, config in models_to_test.items():
        if not os.path.exists(config['weights']):
            print(f"[BỎ QUA] Không tìm thấy file {config['weights']}. Đảm bảo bạn đã chạy xong Test tương ứng.")
            continue
            
        print(f"Đang phân tích: {name}...")
        
        # Load mô hình
        model = build_model(config['type'], freeze_backbone=False)
        model.load_state_dict(torch.load(config['weights'], map_location=device))
        model = model.to(device)
        
        # 1. Kích thước mô hình
        params_mil = count_parameters(model)
        file_size_mb = os.path.getsize(config['weights']) / (1024 * 1024)
        
        # 2. Tốc độ suy luận
        inf_time = measure_inference_time(model, device)
        
        # 3. Điểm chính xác
        acc = get_accuracy(model, val_loader, device)
        
        results.append({
            'Model': name,
            'Accuracy': acc * 100, # Quy ra phần trăm cho dễ nhìn
            'Params (M)': params_mil,
            'Size (MB)': file_size_mb,
            'Inference (ms)': inf_time
        })

    if not results:
        print("Không có mô hình nào được đánh giá. Dừng chương trình.")
        exit()

    # === IN BÁO CÁO RA TERMINAL ===
    print("\n" + "="*85)
    print(f"{'Mô hình':<25} | {'Accuracy (%)':<12} | {'Params (Triệu)':<15} | {'Size (MB)':<10} | {'Suy luận (ms/ảnh)':<15}")
    print("-" * 85)
    for r in results:
        print(f"{r['Model']:<25} | {r['Accuracy']:<12.2f} | {r['Params (M)']:<15.2f} | {r['Size (MB)']:<10.2f} | {r['Inference (ms)']:<15.2f}")
    print("="*85)

    # === VẼ BIỂU ĐỒ BONG BÓNG (BUBBLE CHART) ===
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, r in enumerate(results):
        # Trục X: Thời gian suy luận, Trục Y: Accuracy, Độ lớn bong bóng: Kích thước mô hình (Params)
        plt.scatter(r['Inference (ms)'], r['Accuracy'], 
                    s=r['Params (M)'] * 15, # Hệ số nhân để bong bóng to ra cho dễ nhìn
                    alpha=0.7, color=colors[i%len(colors)], label=r['Model'], edgecolors="black", linewidth=1.5)
        
        # Gắn tên nhãn cạnh bong bóng
        plt.annotate(r['Model'], (r['Inference (ms)'], r['Accuracy']), 
                     textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, fontweight='bold')

    plt.title('Trade-off Hiệu quả: Tốc độ vs Độ chính xác vs Kích thước\n(Độ lớn của bong bóng đại diện cho Số lượng Tham số)', fontsize=14, fontweight='bold')
    plt.xlabel('Thời gian suy luận 1 ảnh (ms) -> Càng thấp (lệch trái) càng tốt', fontsize=12)
    plt.ylabel('Độ chính xác - Accuracy (%) -> Càng cao (lệch trên) càng tốt', fontsize=12)
    plt.legend(loc="lower right", title="Kiến trúc")
    
    os.makedirs('../docs', exist_ok=True)
    save_path = '../docs/efficiency_bubble_chart.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Đã lưu biểu đồ bong bóng cực xịn tại: {save_path}")
    plt.show()