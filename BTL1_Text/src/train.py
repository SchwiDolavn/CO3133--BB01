import numpy as np
import torch
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset import load_and_tokenize_data
from models import build_model_pytorch
from evaluation import evaluate_model, export_results_to_csv, plot_training_history, error_analysis
from losses import focal_loss_pytorch, get_class_weights # <-- Thêm get_class_weights
from efficiency import get_model_size, measure_inference_time_pytorch, apply_quantization 

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    return {
        "accuracy": acc, 
        "f1_macro": f1, 
        "precision": precision, 
        "recall": recall
    }

def train_pipeline_pytorch(scenario_name, model_type='Transformer_BERT', epochs=5, use_focal_loss=False, freeze_backbone=False, use_augmentation=False):
    hf_model_name = "distilbert-base-uncased" if model_type == 'Transformer_DistilBERT' else "bert-base-uncased"

    train_dataset, val_dataset, num_classes, label_encoder, tokenizer = load_and_tokenize_data(
        model_name=hf_model_name,
        augment=use_augmentation 
    )
    
    model = build_model_pytorch(model_type=model_type, num_classes=num_classes, vocab_size=tokenizer.vocab_size)
    
    if freeze_backbone:
        if model_type == 'Transformer_BERT':
            for param in model.bert.parameters():
                param.requires_grad = False
        elif model_type == 'Transformer_DistilBERT':
            for param in model.distilbert.parameters():
                param.requires_grad = False
    
    custom_loss_fn = None
    if use_focal_loss:
        train_labels = train_dataset['labels'].numpy()
        class_weights_tensor = get_class_weights(train_labels)
        custom_loss_fn = focal_loss_pytorch(gamma=2.0, weight=class_weights_tensor)

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

    lr = 2e-5 if 'Transformer' in model_type else 1e-3
    training_args = TrainingArguments(
        output_dir=f"../result/checkpoints_{scenario_name}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=32, 
        per_device_eval_batch_size=32,
        fp16=torch.cuda.is_available(), 
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro", 
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=True
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 
    )
    
    trainer.train()
    
    return trainer, val_dataset, label_encoder, tokenizer