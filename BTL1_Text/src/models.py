import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from losses import focal_loss_pytorch

# 1. Định nghĩa mạng Bi-LSTM bằng PyTorch
class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=32, num_classes=5, dropout=0.5):
        super(BiLSTM_Model, self).__init__()
        # Lớp Embedding (thay thế cho lớp Embedding của Keras)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Lớp Bi-LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Dropout chống Overfitting
        self.dropout = nn.Dropout(dropout)
        # Lớp Dense cuối cùng (Linear trong PyTorch)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None, loss_type='cross_entropy', **kwargs):
        # 1. Đưa text qua lớp Embedding
        embedded = self.embedding(input_ids)
        
        # 2. Đưa qua Bi-LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # 3. Lấy trung bình các từ trong câu (Mean Pooling dựa trên mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
            sum_embeddings = torch.sum(lstm_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(lstm_out, dim=1)

        # 4. Dropout và phân loại
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        # 5. Tự động tính Loss nếu có truyền nhãn (Để dùng được Trainer API)
        loss = None
        if loss_type == 'focal':
            # Sử dụng Focal Loss (đã chuyển sang PyTorch)
            loss_fct = focal_loss_pytorch(gamma=2.0, alpha=0.25)
            loss = loss_fct(logits, labels)
        else:
            # Sử dụng mặc định
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        # Trả về format chuẩn của Hugging Face
        return SequenceClassifierOutput(loss=loss, logits=logits)

# 2. Hàm gọi mô hình (Gom cả 2 vào đây)
def build_model_pytorch(model_type, num_classes, vocab_size=None):
    if model_type == 'Transformer_BERT':
        print("Đang tải mô hình Transformer BERT...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=num_classes
        )
        return model
    
    elif model_type == 'Transformer_DistilBERT':
        print("Đang tải mô hình Transformer DistilBERT (Small)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_classes
        )
        return model
        
    elif model_type == 'RNN_Bi-LSTM':
        print("Đang khởi tạo mạng RNN Bi-LSTM...")
        if vocab_size is None:
            raise ValueError("Cần truyền vocab_size (kích thước từ điển) cho Bi-LSTM.")
        model = BiLSTM_Model(vocab_size=vocab_size, num_classes=num_classes)
        return model
        
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_type}")