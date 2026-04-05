import os
import pickle
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords

# Tải danh sách từ dừng (chỉ cần chạy 1 lần)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text, remove_stop=False):
    # 1. Chuyển về chữ thường
    text = text.lower()
    # 2. Thay thế ký tự @ bằng chữ 'number' để có nghĩa hơn
    text = text.replace("@", "number")
    # 3. Loại bỏ ký tự đặc biệt/dấu câu thừa
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Loại bỏ Stopwords (nếu được yêu cầu - dành cho Bi-LSTM)
    if remove_stop:
        words = text.split()
        text = " ".join([w for w in words if w not in stop_words])
        
    return text

def load_and_tokenize_data(model_name="bert-base-uncased", augment=False):
    print("Đang tải dữ liệu từ Hugging Face Hub (armanc/pubmed-rct20k)...")
    ds = load_dataset("armanc/pubmed-rct20k")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Mã hóa nhãn (Label Encoding)
    label_encoder = LabelEncoder()
    label_encoder.fit(ds['train']['label'])
    num_classes = len(label_encoder.classes_)
    print(f"Đã tìm thấy {num_classes} nhãn: {label_encoder.classes_}")

    os.makedirs('../result', exist_ok=True)
    with open('../result/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # Hàm tiền xử lý: Tokenize text và chuyển nhãn thành số
    def preprocess_function(examples, is_train=True):
        should_remove_stop = "bert" not in model_name.lower()
        cleaned_texts = [clean_text(t, remove_stop=should_remove_stop) for t in examples['text']]
        
        # LOGIC AUGMENTATION: Chỉ kích hoạt khi augment=True VÀ đang xử lý tập Train
        if augment and is_train:
            import random
            aug_texts = []
            for t in cleaned_texts:
                words = t.split()
                if len(words) > 5:
                    words.pop(random.randint(0, len(words)-1))
                aug_texts.append(" ".join(words))
            cleaned_texts = aug_texts

        tokenized = tokenizer(cleaned_texts, padding="max_length", truncation=True, max_length=55)
        tokenized['labels'] = label_encoder.transform(examples['label']).tolist()
        return tokenized

    print("Đang Tokenize tập Train và Validation...")
    
    # THÊM remove_columns=['label'] VÀO 2 DÒNG NÀY
    print("Đang Tokenize tập Train...")
    train_dataset = ds['train'].map(lambda x: preprocess_function(x, is_train=True), batched=True, remove_columns=['label'])

    print("Đang Tokenize tập Validation...")
    val_dataset = ds['validation'].map(lambda x: preprocess_function(x, is_train=False), batched=True, remove_columns=['label'])
    
    # Ép kiểu dữ liệu sang PyTorch Tensors (Giữ nguyên)
    train_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    val_dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    
    return train_dataset, val_dataset, num_classes, label_encoder, tokenizer