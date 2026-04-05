import numpy as np
import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
from dataset import load_and_tokenize_data
from models import build_model_pytorch
from evaluation import evaluate_model

def compute_metrics(eval_pred):
    """Tính độ chính xác trong quá trình huấn luyện"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_pipeline_pytorch(model_type='Transformer_BERT', epochs=3):
    # 1. Tải và xử lý dữ liệu (Dùng chung 1 data pipeline cho cả 2 model)
    train_dataset, val_dataset, num_classes, label_encoder, tokenizer = load_and_tokenize_data()
    
    # 2. Khởi tạo mô hình theo lựa chọn
    model = build_model_pytorch(
        model_type=model_type, 
        num_classes=num_classes, 
        vocab_size=tokenizer.vocab_size # Chỉ Bi-LSTM mới cần dùng tới tham số này
    )
    
    # 3. Cấu hình tham số huấn luyện
    # Đối với Bi-LSTM thường cần learning_rate lớn hơn BERT một chút
    lr = 2e-5 if model_type == 'Transformer_BERT' else 1e-3
    
    training_args = TrainingArguments(
        output_dir=f"../result/{model_type}_checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        
        # 1. GIẢM BATCH SIZE XUỐNG MỘT NỬA
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        
        # 2. BÙ LẠI BATCH SIZE BẰNG GRADIENT ACCUMULATION (8 x 2 = 16)
        gradient_accumulation_steps=2,
        
        # 3. BẬT CHẾ ĐỘ NÉN BỘ NHỚ (GIẢM 50% RAM GPU & CHẠY NHANH HƠN)
        fp16=True, 
        
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        remove_unused_columns=False 
    )
    
    # 4. Bắt đầu huấn luyện với Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print(f"\n🚀 BẮT ĐẦU HUẤN LUYỆN: {model_type}...")
    trainer.train()
    
    # 5. Lưu kết quả
    trainer.save_model(f"../result/{model_type}_best")
    tokenizer.save_pretrained(f"../result/{model_type}_best")
    print("--- HUẤN LUYỆN HOÀN TẤT ---")
    
    return trainer, val_dataset, label_encoder

if __name__ == "__main__":
    import pandas as pd
    from evaluation import export_results_to_csv # Import thêm hàm vừa tạo
    
    models_to_run = ['Transformer_BERT', 'RNN_Bi-LSTM']
    summary_results = [] # Lưu tóm tắt để so sánh 2 model

    for model_name in models_to_run:
        print(f"\n" + "="*50)
        print(f"BẮT ĐẦU QUY TRÌNH CHO: {model_name}")
        print("="*50)
        
        # 1. Huấn luyện
        trainer, val_dataset, label_encoder = train_pipeline_pytorch(model_type=model_name, epochs=3)
        
        # 2. Dự đoán và Đánh giá
        print(f"\n🔍 Đang đánh giá mô hình {model_name}...")
        predictions = trainer.predict(val_dataset)
        y_pred_classes = np.argmax(predictions.predictions, axis=-1)
        y_true = val_dataset['labels'].numpy()
        
        # 3. Xuất file CSV chi tiết từng nhãn
        df_res = export_results_to_csv(y_true, y_pred_classes, label_encoder.classes_, model_name)
        
        # 4. Lưu lại Accuracy tổng quát để so sánh
        acc = accuracy_score(y_true, y_pred_classes)
        summary_results.append({"Model": model_name, "Accuracy": acc})
        
        # Vẽ Confusion Matrix
        evaluate_model(y_true, y_pred_classes, label_encoder.classes_)

    # 5. XUẤT FILE SO SÁNH CHUNG (Cực kỳ quan trọng cho báo cáo)
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv("../result/model_comparison_summary.csv", index=False)
    print("\n📊 ĐÃ TẠO FILE SO SÁNH CHUNG: ../result/model_comparison_summary.csv")