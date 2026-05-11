import os
import time
import torch
import numpy as np
import pandas as pd
import gc
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from train import train_pipeline_pytorch
from models import build_model_pytorch
from evaluation import (
    evaluate_model, export_results_to_csv, plot_training_history, 
    error_analysis, plot_comprehensive_comparison
)
from efficiency import (
    get_model_size, measure_inference_time_pytorch, apply_quantization, plot_efficiency_tradeoffs
)
from explain import explain_prediction_lime, explain_prediction_captum

if __name__ == "__main__":
    os.makedirs('../result', exist_ok=True)

    scenarios = [
        {'name': 'BERT_Freeze', 'model': 'Transformer_BERT', 'freeze': True, 'focal': False, 'aug': False},
        {'name': 'BERT_Full', 'model': 'Transformer_BERT', 'freeze': False, 'focal': False, 'aug': False},
        {'name': 'Bi-LSTM_Full', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': False, 'aug': False},

        {'name': 'BERT_Full_Focal', 'model': 'Transformer_BERT', 'freeze': False, 'focal': True, 'aug': False},
        {'name': 'Bi-LSTM_Focal', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': True, 'aug': False},

        {'name': 'BERT_Augmented', 'model': 'Transformer_BERT', 'freeze': False, 'focal': False, 'aug': True},
        {'name': 'Bi-LSTM_Augmented', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': False, 'aug': True},

        {'name': 'DistilBERT_Small', 'model': 'Transformer_DistilBERT', 'freeze': False, 'focal': False, 'aug': False},
    ]

    summary_results = []
    current_device = "cuda" if torch.cuda.is_available() else "cpu"

    for sc in scenarios:
        print("\n" + "="*70)
        print(f"CASE ĐANG THỰC HIỆN CASE: {sc['name']} CASE")
        print("="*70)

        print(f"[HỆ THỐNG] Mô hình: {sc['model']}")
        if sc.get('freeze'): print("-> CASE Chế độ: Đóng băng Backbone (Feature Extraction).")
        if sc.get('focal'): print("-> CASE Chế độ: Sử dụng Focal Loss (Xử lý lệch nhãn).")
        if sc.get('aug'): print("-> CASE Chế độ: Sử dụng Data Augmentation.")

        start_train = time.time()

        trainer, val_dataset, label_encoder, tokenizer = train_pipeline_pytorch(
            scenario_name=sc['name'],
            model_type=sc['model'],
            epochs=5,
            use_focal_loss=sc['focal'],
            freeze_backbone=sc['freeze'],
            use_augmentation=sc['aug']
        )

        train_duration = time.time() - start_train
        print(f"-> [HOÀN TẤT] Thời gian huấn luyện: {train_duration:.1f} giây!")

        print("\n[HỆ THỐNG] Đang chạy đánh giá trên tập Validation...")
        predictions = trainer.predict(val_dataset)
        y_pred_logits = torch.from_numpy(predictions.predictions)
        y_pred_probs = torch.nn.functional.softmax(y_pred_logits, dim=-1).numpy()
        y_pred_classes = np.argmax(y_pred_probs, axis=-1)
        y_true = np.array(val_dataset['labels'])

        m_size = get_model_size(trainer.model)
        full_loader = DataLoader(val_dataset, batch_size=32)
        
        print(f"Đang đo tốc độ suy luận (Inference time) trên {current_device.upper()}...")
        inf_time = measure_inference_time_pytorch(trainer.model, full_loader, device=current_device)

        print("\n[HỆ THỐNG] Đang giải thích mô hình (XAI)...")
        sample_text = val_dataset['text'][0] # Lấy câu đầu tiên
        
        explain_prediction_lime(sample_text, trainer.model, tokenizer, label_encoder, scenario_name=sc['name'], device=current_device)
        print(f"-> Đã lưu biểu đồ LIME tại: ../result/lime_explanation_{sc['name']}.png")
        
        explain_prediction_captum(sample_text, trainer.model, tokenizer, label_encoder, scenario_name=sc['name'], model_type=sc['model'], device=current_device)
        print(f"-> Đã lưu nội soi Captum tại: ../result/captum_explanation_{sc['name']}.html")

        error_analysis(val_dataset['text'], y_true, y_pred_probs, label_encoder, scenario_name=sc['name'])
        print(f"-> Đã lưu file phân tích lỗi tại: ../result/{sc['name']}_error_analysis.csv")

        export_results_to_csv(y_true, y_pred_classes, label_encoder.classes_, sc['name'])
        print(f"-> Đã lưu Classification Report tại: ../result/{sc['name']}_classification_report.csv")
        
        evaluate_model(y_true, y_pred_classes, label_encoder.classes_, scenario_name=sc['name'])
        print(f"-> Đã lưu Confusion Matrix tại: ../result/{sc['name']}_confusion_matrix.png")
        
        plot_training_history(trainer.state.log_history, scenario_name=sc['name'])
        print(f"-> Đã lưu Biểu đồ Loss/Accuracy tại: ../result/{sc['name']}_training_history.png")

        acc = accuracy_score(y_true, y_pred_classes)
        f1 = f1_score(y_true, y_pred_classes, average='macro')
        
        summary_results.append({
            "Scenario": sc['name'],
            "Accuracy": f"{acc*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
            "Train Time (s)": f"{train_duration:.1f}",
            "Inference Time (s)": f"{inf_time:.4f}",
            "Model Size (MB)": f"{m_size:.2f}"
        })

        print("\nCASE Đang dọn dẹp VRAM để chuẩn bị cho CASE tiếp theo...")
        del trainer
        del predictions
        del y_pred_logits
        del full_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[HỆ THỐNG] Đang thực hiện Quantization (INT8) để minh họa Efficiency...")
    bert_base_model = build_model_pytorch('Transformer_BERT', num_classes=len(label_encoder.classes_), vocab_size=tokenizer.vocab_size)
    q_model = apply_quantization(bert_base_model)
    print(f"-> Kích thước BERT ban đầu: {get_model_size(bert_base_model):.2f} MB")
    print(f"-> Kích thước sau khi nén: {get_model_size(q_model):.2f} MB")

    print("\n[HỆ THỐNG] Đang vẽ các biểu đồ so sánh tổng hợp...")
    plot_efficiency_tradeoffs(summary_results)
    plot_comprehensive_comparison(summary_results)

    df_final = pd.DataFrame(summary_results)
    df_final.to_csv("../result/comprehensive_comparison_report.csv", index=False)

    print("\n" + "V" * 70)
    print(" HOÀN TẤT TOÀN BỘ QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ 🎉")
    print(" KẾT QUẢ TỔNG HỢP NẰM TẠI: ../result/comprehensive_comparison_report.csv")
    print("V" * 70)