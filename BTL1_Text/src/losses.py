import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_class_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return torch.tensor(weights, dtype=torch.float)

def focal_loss_pytorch(gamma=2.0, weight=None):
    def focal_loss_fn(logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**gamma * ce_loss
        if weight is not None:
            weight_device = weight.to(labels.device)
            alpha_t = weight_device.gather(0, labels)
            loss = alpha_t * loss
            
        return loss.mean()
        
    return focal_loss_fn