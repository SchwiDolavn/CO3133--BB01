import numpy as np
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

def explain_prediction_lime(text, model, label_encoder, model_type='RNN_Bi-LSTM'):
    """
    Sử dụng LIME để giải thích mô hình NLP.
    Lưu ý: model phải có hàm predict nhận vào list text và trả ra xác suất (softmax).
    """
    print(f"\n--- ĐANG PHÂN TÍCH LIME CHO CÂU ---")
    print(text)
    
    class_names = label_encoder.classes_
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Tạo hàm wrapper dự đoán cho LIME
    def predictor(texts):
        # LIME truyền vào list các chuỗi, ta cần chuyển thành numpy array
        texts = np.array(texts)
        if 'RNN' in model_type:
            preds = model.predict(texts)
            return preds
        elif 'Transformer' in model_type:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # Encode text
            encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors='tf')
            # Predict
            logits = model(dict(encodings)).logits
            # Chuyển logits thành xác suất bằng softmax
            preds = tf.nn.softmax(logits, axis=-1).numpy()
            return preds
    
    # Giải thích dự đoán
    exp = explainer.explain_instance(text, predictor, num_features=10, num_samples=500)
    
    # In ra kết quả dạng text
    print("\n[LIME] Các từ ảnh hưởng nhất:")
    print(exp.as_list())
    
    # Lưu biểu đồ ra file
    fig = exp.as_pyplot_figure()
    plt.title(f"Giải thích LIME - Dự đoán: {class_names[np.argmax(predictor([text])[0])]}")
    plt.tight_layout()
    plt.savefig('../result/lime_explanation.png')
    print("Đã lưu biểu đồ giải thích tại ../result/lime_explanation.png")