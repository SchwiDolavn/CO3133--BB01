# Bài Tập Lớn 1: Học Sâu và Ứng Dụng (CO3133) - Phân loại Văn bản

## 📝 Giới thiệu bài toán
[cite_start]Dự án này thực hiện bài toán phân loại văn bản trên tập dữ liệu **PubMed RCT 20k** (tóm tắt các bài báo y khoa)[cite: 191, 194]. [cite_start]Mục tiêu là phân loại các câu văn vào 5 nhãn: `Background`, `Objective`, `Methods`, `Results`, `Conclusions`[cite: 200].

Nhóm tập trung so sánh giữa hai họ mô hình:
* [cite_start]**RNN (Bi-LSTM):** Xử lý tuần tự dựa trên cơ chế bộ nhớ[cite: 380, 400].
* [cite_start]**Transformer (BERT/DistilBERT):** Dựa trên cơ chế chú ý (Attention) mạnh mẽ[cite: 425, 428, 591].

## 📂 Cấu trúc thư mục
```text
BTL1_Text/
├── data/               # Chứa dữ liệu thô (nếu có)
├── notebook/           # Các file Jupyter Notebook thử nghiệm
├── result/             # Kết quả: Biểu đồ, Confusion Matrix, báo cáo CSV
└── src/                # Mã nguồn chính của dự án
    ├── dataset.py      # Tiền xử lý, Augmentation và Dataloader [cite: 351]
    ├── models.py       # Định nghĩa kiến trúc Bi-LSTM và Transformer [cite: 379]
    ├── train.py        # Pipeline huấn luyện sử dụng Hugging Face Trainer
    ├── evaluation.py   # Các hàm đánh giá (Accuracy, F1, Error Analysis) [cite: 64]
    ├── losses.py       # Triển khai Focal Loss cho dữ liệu mất cân bằng [cite: 85]
    ├── efficiency.py   # Đo kích thước, tốc độ và nén mô hình (Quantization) [cite: 77]
    ├── explain.py      # Giải thích mô hình bằng kỹ thuật LIME [cite: 71]
    └── run.py          # Script thực hiện toàn bộ 8 kịch bản so sánh

## 🛠️ Báo cáo chi tiết công việc các Module
1. src/dataset.py (Tiền xử lý và Tăng cường)
Làm sạch dữ liệu chuyên sâu: Chuyển văn bản về chữ thường, loại bỏ dấu câu thừa và đặc biệt là quy chuẩn ký tự @ thành chữ number để bảo toàn giá trị ngữ nghĩa của số liệu y khoa.Xử lý nhiễu theo kiến trúc: Loại bỏ Stopwords cho Bi-LSTM để giảm nhiễu tuần tự, nhưng giữ nguyên cho BERT để phục vụ cơ chế Attention hai chiều.Tăng cường dữ liệu (Augmentation): Áp dụng kỹ thuật xóa ngẫu nhiên từ (Random Word Deletion) trong câu huấn luyện nhằm tăng tính bền bỉ (Robustness) và chống học vẹt từ khóa.Mã hóa nhãn: Sử dụng LabelEncoder để chuẩn hóa 5 nhãn y khoa thành các chỉ số từ 0-4.

2. src/models.py (Kiến trúc mô hình)
Kiến trúc Bi-LSTM tùy chỉnh: Xây dựng mạng RNN hai chiều kết hợp lớp Embedding và Mean Pooling để trích xuất đặc trưng toàn câu.Transformer Transfer Learning: Tích hợp bert-base-uncased (110 triệu tham số) và distilbert-base-uncased (phiên bản rút gọn 40%) để so sánh hiệu năng thực tế.

3. src/losses.py & src/train.py (Huấn luyện mở rộng)
Focal Loss: Triển khai hàm mất mát Focal Loss thay thế cho CrossEntropy thông thường nhằm ép mô hình tập trung vào các lớp thiểu số khó nhận diện (như Objective).Cơ chế đóng băng (Freeze): Công tắc freeze_backbone cho phép đánh giá sự khác biệt giữa việc chỉ trích xuất đặc trưng (Feature Extraction) và tinh chỉnh toàn bộ mạng (Fine-tuning).

4. src/evaluation.py & src/explain.py (Phân tích định tính/định lượng)
Phân tích lỗi (Error Analysis): Tự động trích xuất các mẫu bị đoán sai ra file CSV kèm độ tự tin (Confidence) để tìm ra ranh giới nhầm lẫn giữa các nhãn y khoa.Giải thích LIME: Trực quan hóa trọng số tích cực/tiêu cực của các từ khóa ảnh hưởng đến quyết định phân loại, giúp hiểu rõ cơ chế "nhìn" văn bản của mô hình.

5. src/efficiency.py (Đánh giá tài nguyên)
Nén mô hình (Quantization): Áp dụng INT8 Dynamic Quantization để giảm dung lượng file trọng số mà vẫn giữ vững độ chính xác.Đo tốc độ suy luận: Tính toán thời gian phản hồi thực tế (Inference time) trên mỗi câu văn.

## 🧪 Chiến lược thực nghiệm (Testing Strategy)
Nhóm thực hiện chạy tự động 8 kịch bản (Test cases) trong run.py để lấy điểm mở rộng tối đa:

Nhóm Chiến lược Fine-tune (Test 01, 02): So sánh giữa việc chỉ huấn luyện lớp Classifier (BERT_Freeze) và huấn luyện toàn bộ mạng (BERT_Full). Kỳ vọng chứng minh BERT đóng băng không hiệu quả trên dữ liệu chuyên ngành y tế chuyên sâu.

Nhóm Kiến trúc đối chứng (Test 03, 08): So sánh sức mạnh giữa Bi-LSTM (RNN) và BERT (Transformer). Đánh giá khả năng ghi nhớ dài hạn vs cơ chế Attention toàn cục.

Nhóm Xử lý mất cân bằng (Test 04, 05): Áp dụng Focal Loss cho cả hai kiến trúc để cải thiện chỉ số F1-score của nhãn thiểu số Objective. 

Nhóm Hiệu quả phần cứng (Test 06): Thử nghiệm với DistilBERT_Small để chứng minh sự đánh đổi (Trade-off) giữa độ chính xác và tốc độ/kích thước mô hình. Nhóm Tăng cường & Robustness (Test 07): Bật kỹ thuật xóa từ ngẫu nhiên để kiểm tra xem mô hình có bị sụt giảm độ chính xác hay sẽ trở nên bền bỉ hơn với dữ liệu lỗi. 