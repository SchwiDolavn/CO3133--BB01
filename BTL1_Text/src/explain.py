import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

def explain_prediction_lime(text, model, tokenizer, label_encoder, scenario_name="model", device="cpu"):
    """
    Sử dụng LIME để giải thích mô hình NLP trên PyTorch.
    """
    model.to(device)
    model.eval()
    class_names = list(label_encoder.classes_)
    explainer = LimeTextExplainer(class_names=class_names)
    
    def predictor(texts):
        # --- BẢN VÁ QUAN TRỌNG: Chia batch để tránh tràn RAM GPU ---
        batch_size = 32  # Xử lý 32 câu mỗi lần thay vì 500 câu cùng lúc
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=55)
            inputs = {k: v.to(device) for k, v in inputs.items()} 
        
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                
        # Nối kết quả của các batch lại thành 1 mảng numpy duy nhất cho LIME
        return np.vstack(all_probs)
    
    # Giải thích dự đoán cho 1 câu cụ thể
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=500)
    
    # Hiển thị và lưu kết quả
    print("\n[LIME] Các từ ảnh hưởng nhất đến quyết định của mô hình:")
    for feature, weight in exp.as_list():
        print(f" - {feature}: {weight:.4f}")
    
    # Vẽ biểu đồ
    fig = exp.as_pyplot_figure()
    plt.title(f"Giải thích LIME - Kịch bản: {scenario_name}") # Thêm tên kịch bản vào title cho rõ ràng
    plt.tight_layout()
    file_name = f'../result/lime_explanation_{scenario_name}.png'
    plt.savefig(file_name)
    plt.close() 
    print(f"✅ Đã lưu biểu đồ giải thích tại {file_name}")