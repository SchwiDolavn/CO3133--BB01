import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import thư viện Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import từ project của bạn
from models import build_model
from dataset import get_dataloaders

def get_gradcam_images(model, target_layer, dataloader, class_names, device, num_images=8):
    """
    Chạy Grad-CAM trên một số ảnh ngẫu nhiên và vẽ bản đồ nhiệt.
    """
    model.eval()
    
    # Khởi tạo đối tượng GradCAM
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    images_processed = 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    print("Đang tạo bản đồ nhiệt Grad-CAM...")

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        
        # Lấy dự đoán của mô hình
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        for i in range(inputs.size(0)):
            if images_processed >= num_images:
                break
                
            input_tensor = inputs[i].unsqueeze(0) # Thêm batch dimension (1, C, H, W)
            true_label = labels[i].item()
            pred_label = preds[i].item()
            
            # 1. Tạo mask Grad-CAM cho class mà mô hình đã dự đoán
            targets = [ClassifierOutputTarget(pred_label)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            # 2. Giải chuẩn hóa ảnh gốc để hiển thị
            img_tensor = inputs[i].cpu().clone()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_unnorm = img_tensor * std + mean
            
            # Chuyển ảnh về định dạng numpy float32 trong khoảng [0, 1] cho hàm show_cam_on_image
            rgb_img = img_unnorm.permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)
            
            # 3. Phủ bản đồ nhiệt lên ảnh gốc
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            # 4. Hiển thị lên lưới đồ thị
            axes[images_processed].imshow(cam_image)
            
            # Đổi màu text nếu đoán sai (để dễ nhận biết)
            color = 'green' if true_label == pred_label else 'red'
            title = f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}"
            axes[images_processed].set_title(title, color=color, fontweight='bold')
            axes[images_processed].axis('off')
            
            images_processed += 1
            
        if images_processed >= num_images:
            break

    plt.suptitle("Giải thích Mô hình bằng Grad-CAM (Vùng màu đỏ là nơi mô hình chú ý)", fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('../docs', exist_ok=True)
    save_path = '../docs/gradcam_analysis.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n=> Đã lưu ảnh phân tích Grad-CAM tại: {save_path}")
    plt.show()

if __name__ == '__main__':
    DATA_DIR = '../dataset/bloodcells_dataset'
    WEIGHTS_PATH = '../weights/best_09_ResNet_Ultimate.pth' # Dùng mô hình CNN xịn nhất của bạn
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataloader (shuffle=True để mỗi lần chạy lấy ra các ảnh khác nhau)
    _, val_loader, class_names, _ = get_dataloaders(DATA_DIR, batch_size=16)
    
    # 2. Khởi tạo mô hình ResNet18 và load trọng số
    model = build_model('resnet18', freeze_backbone=False)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model = model.to(device)
        
        # 3. Chọn layer cuối cùng của Convolutional block để xem đặc trưng
        # Đối với ResNet18, layer tốt nhất để visualize là layer4[-1]
        target_layer = model.layer4[-1]
        
        # 4. Chạy Grad-CAM
        get_gradcam_images(model, target_layer, val_loader, class_names, device)
    else:
        print(f"[LỖI] Không tìm thấy file trọng số tại {WEIGHTS_PATH}")