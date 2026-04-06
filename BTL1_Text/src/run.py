import os
import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from train import train_pipeline_pytorch
from models import build_model_pytorch
from evaluation import evaluate_model, export_results_to_csv, plot_training_history, error_analysis
from efficiency import get_model_size, measure_inference_time_pytorch, apply_quantization
from explain import explain_prediction_lime

if __name__ == "__main__":
    # Đảm bảo thư mục kết quả tồn tại
    os.makedirs('../result', exist_ok=True)

    # ĐỊNH NGHĨA CÁC KỊCH BẢN ĐỂ LẤY ĐIỂM MỞ RỘNG TỐI ĐA
    scenarios = [
        # 1. NHÓM: CHIẾN LƯỢC FINE-TUNE
        # (Bi-LSTM mặc định là Full, không có khái niệm Freeze)
        {'name': 'BERT_Freeze', 'model': 'Transformer_BERT', 'freeze': True, 'focal': False, 'aug': False},
        {'name': 'BERT_Full', 'model': 'Transformer_BERT', 'freeze': False, 'focal': False, 'aug': False},
        {'name': 'Bi-LSTM_Full', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': False, 'aug': False}, # Baseline cho RNN

        # 2. NHÓM: DỮ LIỆU MẤT CÂN BẰNG (Focal Loss)
        {'name': 'BERT_Full_Focal', 'model': 'Transformer_BERT', 'freeze': False, 'focal': True, 'aug': False},
        {'name': 'Bi-LSTM_Focal', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': True, 'aug': False},

        # 3. NHÓM: AUGMENTATION & ROBUSTNESS
        {'name': 'BERT_Augmented', 'model': 'Transformer_BERT', 'freeze': False, 'focal': False, 'aug': True},
        {'name': 'Bi-LSTM_Augmented', 'model': 'RNN_Bi-LSTM', 'freeze': False, 'focal': False, 'aug': True},

        # 4. NHÓM: HIỆU QUẢ MÔ HÌNH (Mô hình nhỏ)
        {'name': 'DistilBERT_Small', 'model': 'Transformer_DistilBERT', 'freeze': False, 'focal': False, 'aug': False},
    ]

    summary_results = []

    for sc in scenarios:
        print(f"\n" + "="*60)
        print(f" ĐANG THỰC HIỆN KỊCH BẢN: {sc['name']} ")
        print("="*60)

        start_train = time.time()

        # 1. Huấn luyện theo kịch bản (Truyền toàn bộ biến điều khiển vào pipeline)
        trainer, val_dataset, label_encoder, tokenizer = train_pipeline_pytorch(
            scenario_name=sc['name'],
            model_type=sc['model'],
            epochs=3,
            use_focal_loss=sc['focal'],
            freeze_backbone=sc['freeze'],
            use_augmentation=sc['aug']
        )

        train_duration = time.time() - start_train

        # 2. Dự đoán và trích xuất xác suất (cho Error Analysis/LIME)
        predictions = trainer.predict(val_dataset)
        y_pred_logits = torch.from_numpy(predictions.predictions)
        y_pred_probs = torch.nn.functional.softmax(y_pred_logits, dim=-1).numpy()
        y_pred_classes = np.argmax(y_pred_probs, axis=-1)
        y_true = np.array(val_dataset['labels'])

        # 3. Đánh giá Hiệu quả (Efficiency)
        m_size = get_model_size(trainer.model)

        # SỬA Ở ĐÂY: Chỉ lấy 100 câu đầu tiên để đo tốc độ cho siêu nhanh
        subset_val = val_dataset.select(range(100))
        temp_loader = DataLoader(subset_val, batch_size=1)

        # Đo tốc độ trên 100 câu
        print("⏱️ Đang đo tốc độ suy luận (Inference time) trên CPU...")
        inf_time = measure_inference_time_pytorch(trainer.model, temp_loader, device="cpu")

        # 4. Giải thích mô hình (Interpretability - chỉ chạy cho câu đầu tiên)
        sample_text = val_dataset['text'][0]
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        explain_prediction_lime(sample_text, trainer.model, tokenizer, label_encoder, scenario_name=sc['name'], device=current_device)

        # 5. Phân tích lỗi (Error Analysis)
        error_analysis(val_dataset['text'], y_true, y_pred_probs, label_encoder, scenario_name=sc['name'])

        # 6. Xuất báo cáo và biểu đồ cho kịch bản hiện tại
        export_results_to_csv(y_true, y_pred_classes, label_encoder.classes_, sc['name'])
        evaluate_model(y_true, y_pred_classes, label_encoder.classes_, scenario_name=sc['name'])
        plot_training_history(trainer.state.log_history, scenario_name=sc['name'])

        # 7. Lưu kết quả vào bảng tổng hợp
        acc = accuracy_score(y_true, y_pred_classes)
        summary_results.append({
            "Scenario": sc['name'],
            "Accuracy": f"{acc*100:.2f}%",
            "Train Time (s)": f"{train_duration:.1f}",
            "Inference Time (s)": f"{inf_time:.4f}",
            "Model Size (MB)": f"{m_size:.2f}"
        })

        # --- THÊM ĐOẠN NÀY ĐỂ TRÁNH CRASH TRÀN RAM/GPU ---
        print("🧹 Đang dọn dẹp bộ nhớ để chuẩn bị cho kịch bản tiếp theo...")
        del trainer
        del predictions
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- SAU KHI CHẠY XONG CÁC KỊCH BẢN ---

    # 8. MỞ RỘNG: Thử nghiệm Quantization cho mô hình BERT cuối cùng
    print("\n🛠️ Đang thực hiện Quantization (Nén mô hình) để minh họa Efficiency...")
    # Khởi tạo lại một mô hình BERT để minh họa việc nén
    bert_base_model = build_model_pytorch('Transformer_BERT', num_classes=len(label_encoder.classes_), vocab_size=tokenizer.vocab_size)
    q_model = apply_quantization(bert_base_model)
    print(f"✅ Kích thước BERT ban đầu: {get_model_size(bert_base_model):.2f} MB")
    print(f"✅ Kích thước sau khi nén (INT8): {get_model_size(q_model):.2f} MB")

    # 9. Xuất báo cáo so sánh tổng thể
    df_final = pd.DataFrame(summary_results)
    df_final.to_csv("../result/comprehensive_comparison_report.csv", index=False)

    print("\n" + "V" * 60)
    print(" ✅ HOÀN TẤT TOÀN BỘ QUY TRÌNH MỞ RỘNG.")
    print(" ✅ KẾT QUẢ TỔNG HỢP TẠI: ../result/comprehensive_comparison_report.csv")
    print("V" * 60)