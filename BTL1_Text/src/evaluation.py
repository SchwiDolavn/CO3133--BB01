import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Biểu đồ Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Biểu đồ Accuracy')
    ax2.legend()
    
    plt.show()

def evaluate_model(y_true, y_pred, classes):
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    plt.show()

def error_analysis(texts, y_true, y_pred_probs, label_encoder, top_n=5):
    """
    Trích xuất các câu dự đoán sai nhưng mô hình lại có độ tự tin rất cao.
    texts: List các câu (string)
    y_true: Nhãn thực tế (số nguyên)
    y_pred_probs: Xác suất dự đoán từ model.predict()
    """
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)
    
    # Tìm các vị trí dự đoán sai
    errors_idx = np.where(y_pred_classes != y_true)[0]
    
    # Tạo DataFrame để phân tích
    error_data = []
    for idx in errors_idx:
        error_data.append({
            'Text': texts[idx],
            'True_Label': label_encoder.inverse_transform([y_true[idx]])[0],
            'Predicted_Label': label_encoder.inverse_transform([y_pred_classes[idx]])[0],
            'Confidence': confidences[idx]
        })
        
    df_errors = pd.DataFrame(error_data)
    
    # Sắp xếp theo độ tự tin giảm dần (Sai nhưng "cãi cố")
    df_errors = df_errors.sort_values(by='Confidence', ascending=False)
    
    print(f"\n--- PHÂN TÍCH LỖI (TOP {top_n} CÂU DỰ ĐOÁN SAI VỚI ĐỘ TỰ TIN CAO NHẤT) ---")
    for i, row in df_errors.head(top_n).iterrows():
        print(f"\n[Confidence: {row['Confidence']:.4f}]")
        print(f"Thực tế: {row['True_Label']} | Dự đoán: {row['Predicted_Label']}")
        print(f"Văn bản: {row['Text'][:200]}...") # In 200 ký tự đầu
        
    # Lưu file CSV để làm báo cáo
    df_errors.to_csv('../result/error_analysis.csv', index=False)
    print("\nĐã xuất toàn bộ phân tích lỗi ra ../result/error_analysis.csv")

def export_results_to_csv(y_true, y_pred, classes, model_name):
    """Xuất báo cáo chi tiết ra file CSV để làm bảng biểu"""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    file_path = f"../result/{model_name}_classification_report.csv"
    df_report.to_csv(file_path)
    print(f"✅ Đã xuất kết quả chi tiết ra: {file_path}")
    return df_report