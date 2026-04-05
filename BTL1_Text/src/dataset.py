import os
import pickle
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

def load_and_tokenize_data(model_name="bert-base-uncased"):
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
    def preprocess_function(examples):
        tokenized = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
        tokenized['labels'] = label_encoder.transform(examples['label']).tolist()
        return tokenized

    print("Đang Tokenize tập Train và Validation...")
    # Dùng map để xử lý song song siêu tốc
    tokenized_datasets = ds.map(preprocess_function, batched=True, remove_columns=ds['train'].column_names)
    
    # Ép kiểu dữ liệu sang PyTorch Tensors
    tokenized_datasets.set_format("torch")

    return tokenized_datasets['train'], tokenized_datasets['validation'], num_classes, label_encoder, tokenizer