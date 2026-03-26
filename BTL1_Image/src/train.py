import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import copy

from dataset import get_dataloaders
from models import build_model
from losses import FocalLoss

def get_optimizer_layer_wise(model, model_type, base_lr=1e-4):
    """Set Learning Rate khác nhau cho Backbone và Classifier"""
    if model_type == 'resnet18':
        ignored_params = list(map(id, model.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = optim.Adam([
            {'params': base_params, 'lr': base_lr},       # Backbone học chậm
            {'params': model.fc.parameters(), 'lr': base_lr * 10} # Classifier học nhanh
        ])
    elif model_type == 'vit_b_16':
        ignored_params = list(map(id, model.heads.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = optim.Adam([
            {'params': base_params, 'lr': base_lr},
            {'params': model.heads.parameters(), 'lr': base_lr * 10}
        ])
    else: 
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    return optimizer

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_name="model", patience=5):
    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN {model_name.upper()} ===")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0 
    
    start_time = time.time() 

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader

            running_loss, running_corrects = 0.0, 0
            for inputs, labels in tqdm(dataloader, desc=f"{phase.capitalize()}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train': scheduler.step()

            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    os.makedirs('../weights', exist_ok=True)
                    torch.save(model.state_dict(), f'../weights/best_{model_name}.pth')
                    epochs_no_improve = 0 
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("-> Early Stopping!")
            break

    total_time = time.time() - start_time
    print(f'\nHoàn tất {model_name}. Best Val Acc: {best_acc:4f}. Thời gian train: {total_time//60:.0f} phút {total_time%60:.0f} giây')
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    DATA_DIR = '../dataset/bloodcells_dataset'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_EPOCHS = 20 

    experiments = [
        {"name": "01_ResNet_Freeze", "model": "resnet18", "strategy": "freeze", "aug": False, "focal": False},
        {"name": "02_ResNet_Full", "model": "resnet18", "strategy": "full", "aug": False, "focal": False},
        {"name": "03_ResNet_LayerWise", "model": "resnet18", "strategy": "layer_wise", "aug": False, "focal": False},        
        {"name": "04_ViT_Freeze", "model": "vit_b_16", "strategy": "freeze", "aug": False, "focal": False},
        {"name": "05_ViT_LayerWise", "model": "vit_b_16", "strategy": "layer_wise", "aug": False, "focal": False},
        {"name": "06_MobileNet_LayerWise", "model": "mobilenet_v3", "strategy": "layer_wise", "aug": False, "focal": False},
        {"name": "07_ResNet_Augment", "model": "resnet18", "strategy": "layer_wise", "aug": True, "focal": False},
        {"name": "08_ResNet_FocalLoss", "model": "resnet18", "strategy": "layer_wise", "aug": False, "focal": True},
        {"name": "09_ResNet_Ultimate", "model": "resnet18", "strategy": "layer_wise", "aug": True, "focal": True},
        {"name": "10_ViT_Ultimate", "model": "vit_b_16", "strategy": "layer_wise", "aug": True, "focal": True},
    ]

    

    for i, exp in enumerate(experiments):
        print("\n" + "="*70)
        print(f"THỬ NGHIỆM {i+1}/{len(experiments)}: {exp['name']}")
        print(f"Cấu hình: Model={exp['model']} | Strategy={exp['strategy']} | Augment={exp['aug']} | FocalLoss={exp['focal']}")
        print("="*70)

        # 1. Khởi tạo Dataloader (Gọi lại mỗi lần vì tham số Augment có thể thay đổi)
        train_loader, val_loader, class_names, class_weights = get_dataloaders(
            DATA_DIR, batch_size=32, use_advanced_aug=exp['aug']
        )

        # 2. Khởi tạo Hàm Loss
        criterion = FocalLoss() if exp['focal'] else nn.CrossEntropyLoss(weight=class_weights.to(device))

        # 3. Khởi tạo Mô hình
        is_frozen = True if exp['strategy'] == 'freeze' else False
        model = build_model(exp['model'], freeze_backbone=is_frozen)

        # 4. Cài đặt Tốc độ học (Optimizer)
        base_lr = 5e-5 if exp['model'] == 'vit_b_16' else 1e-4

        if exp['strategy'] == 'layer_wise':
            optimizer = get_optimizer_layer_wise(model, exp['model'], base_lr=base_lr)
        else:
            if is_frozen:
                if exp['model'] == 'resnet18': params_to_update = model.fc.parameters()
                elif exp['model'] == 'vit_b_16': params_to_update = model.heads.parameters()
                else: params_to_update = model.classifier.parameters()
            else:
                params_to_update = model.parameters()
                
            optimizer = optim.Adam(params_to_update, lr=base_lr)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        # 5. Đưa vào vòng lặp Huấn luyện
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                    num_epochs=NUM_EPOCHS, device=device, model_name=exp['name'], patience=5)

    print("\n ĐÃ HOÀN THÀNH 10 TEST!")