import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class BiLSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=32, num_classes=5, dropout=0.5):
        super(BiLSTM_Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embedded = self.embedding(input_ids)
        
        lstm_out, _ = self.lstm(embedded)
        
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
            sum_embeddings = torch.sum(lstm_out * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(lstm_out, dim=1)

        pooled = self.dropout(pooled)
        logits = self.fc(pooled)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.fc.out_features), labels.view(-1))
        return SequenceClassifierOutput(loss=loss, logits=logits)

def build_model_pytorch(model_type, num_classes, vocab_size=None):
    if model_type == 'Transformer_BERT':
        print("Đang tải mô hình Transformer BERT")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            num_labels=num_classes
        )
        return model
    
    elif model_type == 'Transformer_DistilBERT':
        print("Đang tải mô hình Transformer DistilBERT (Small)")
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=num_classes
        )
        return model
        
    elif model_type == 'RNN_Bi-LSTM':
        print("Đang khởi tạo mạng RNN Bi-LSTM")
        if vocab_size is None:
            raise ValueError("Cần truyền vocab_size (kích thước từ điển) cho Bi-LSTM.")
        model = BiLSTM_Model(vocab_size=vocab_size, num_classes=num_classes)
        return model
        
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_type}")