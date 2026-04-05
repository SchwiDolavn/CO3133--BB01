import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_class_weights(y_train):
    """Tính toán trọng số và trả về dạng PyTorch Tensor"""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    # Chuyển sang Tensor để PyTorch có thể tính toán được
    return torch.tensor(weights, dtype=torch.float)

def focal_loss_pytorch(gamma=2.0, alpha=0.25):
    """Bản PyTorch để khớp với models.py"""
    def focal_loss_fn(logits, labels):
        # Tính Cross Entropy mặc định trước
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        
        # Tính xác suất p_t
        p_t = torch.exp(-ce_loss)
        
        # Công thức Focal Loss: loss = alpha * (1-p_t)^gamma * ce_loss
        loss = alpha * (1 - p_t)**gamma * ce_loss
        return loss.mean()
        
    return focal_loss_fn