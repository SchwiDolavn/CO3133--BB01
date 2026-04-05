import numpy as np
import torch
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score

# Import các module custom
from dataset import load_and_tokenize_data
from models import build_model_pytorch
from evaluation import (
    evaluate_model, export_results_to_csv, plot_training_history, 
    error_analysis
)
from losses import focal_loss_pytorch 
from efficiency import get_model_size, measure_inference_time_pytorch, apply_quantization 
from explain import explain_prediction_lime 

def compute_metrics(eval_pred):
    """Tính toán Accuracy cho Trainer"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_pipeline_pytorch(scenario_name, model_type='Transformer_BERT', epochs=3, use_focal_loss=False, freeze_backbone=False, use_augmentation=False):    # Xác định đúng bộ Tokenizer dựa trên loại mô hình
    hf_model_name = "distilbert-base-uncased" if model_type == 'Transformer_DistilBERT' else "bert-base-uncased"

    # 1. Tải và xử lý dữ liệu (Truyền tham số augment vào dataset)
    train_dataset, val_dataset, num_classes, label_encoder, tokenizer = load_and_tokenize_data(
        model_name=hf_model_name,
        augment=use_augmentation # Đảm bảo file dataset.py của bạn đã nhận tham số này
    )
    
    # 2. Khởi tạo mô hình
    model = build_model_pytorch(
        model_type=model_type, 
        num_classes=num_classes, 
        vocab_size=tokenizer.vocab_size
    )
    
    # --- LOGIC FREEZE BACKBONE (MỞ RỘNG) ---
    if freeze_backbone:
        print(f"❄️ Đang đóng băng Backbone cho {model_type} (Feature Extraction mode)...")
        if model_type == 'Transformer_BERT':
            for param in model.bert.parameters():
                param.requires_grad = False
        elif model_type == 'Transformer_DistilBERT':
            for param in model.distilbert.parameters():
                param.requires_grad = False
    
    # --- LOGIC FOCAL LOSS (MỞ RỘNG: DỮ LIỆU LỆCH LỚP) ---
    custom_loss_fn = None
    if use_focal_loss:
        print(f"🔥 Đang kích hoạt Focal Loss cho {model_type}...")
        custom_loss_fn = focal_loss_pytorch(gamma=2.0, alpha=0.25)

    # 3. Định nghĩa Custom Trainer để hỗ trợ Custom Loss
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if custom_loss_fn is not None:
                loss = custom_loss_fn(logits, labels)
            else:
                loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

    # 4. Cấu hình Training Arguments
    lr = 2e-5 if 'Transformer' in model_type else 1e-3
    training_args = TrainingArguments(
        output_dir=f"../result/checkpoints_{scenario_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=16,
        fp16=torch.cuda.is_available(), # Bật nếu có GPU
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=True
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN: {model_type}...")
    trainer.train()
    
    return trainer, val_dataset, label_encoder, tokenizer

