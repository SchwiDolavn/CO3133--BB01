import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_type, num_classes=8, freeze_backbone=False):
    if model_type == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters(): param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters(): param.requires_grad = False
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif model_type == 'mobilenet_v3': # Mô hình siêu nhẹ, suy luận nhanh
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for param in model.parameters(): param.requires_grad = False
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    return model

class EnsembleModel(nn.Module):
    """Kết hợp dự đoán của 2 mô hình bằng cách lấy trung bình cộng logits"""
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        return (outA + outB) / 2.0