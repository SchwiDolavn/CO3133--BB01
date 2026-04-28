# CO3133 - Deep Learning (BB01)

## Giới thiệu

Dự án này là bài tập lớn số 1 của môn học **CO3133 - Học Sâu và Ứng Dụng** (Deep Learning). Nhóm thực hiện bài toán phân loại trên ba loại dữ liệu: ảnh, văn bản và đa phương thức (ảnh + văn bản).

## Thông tin 
| **Giảng viên** | TS. Lê Thành Sách |
| **Mã nhóm** | BB01 |
| **Số thành viên** | 3 sinh viên |
| **Tên thành viên** | **MSSV** |
|Đinh Ngụy Nguyệt Hà|2352286|
|Nguyễn Đăng Khánh|2311512|
|Hà Anh Tuấn|2252867|

## Các bài tập lớn

### 1. Bài tập lớn 1 - Phân loại Ảnh (Image Classification)
- **Dataset**: Blood Cells Classification (8 lớp tế bào máu)
- **Mục tiêu**: So sánh CNN (ResNet18) vs Vision Transformer (ViT)
- **Kỹ thuật**: Transfer Learning, Data Augmentation, Focal Loss
- **Trang web**: [Image.html](BTL1_Image/Image.html)

### 2. Bài tập lớn 1 - Phân loại Văn bản (Text Classification)
- **Dataset**: PubMed RCT 20k (5 lớp: Background, Objective, Methods, Results, Conclusions)
- **Mục tiêu**: So sánh RNN (Bi-LSTM) vs Transformer (BERT)
- **Kỹ thuật**: Transfer Learning, Focal Loss, Augmentation, Quantization
- **Trang web**: [Text.html](BTL1_Text/Text.html)

### 3. Bài tập lớn 1 - Phân loại Đa phương thức (Multimodal Classification)
- **Dataset**: ROCO Radiology (ảnh X-quang + caption)
- **Mục tiêu**: So sánh Zero-shot vs Few-shot classification
- **Kỹ thuật**: CLIP, Prompt Engineering
- **Trang web**: [ImageText.html](BTL1_ImageText/ImageText.html)

## Cấu trúc dự án

```
CO3133--BB01/
├── index.html              # Trang chủ Landing Page
├── README.md               # File README này
├── BTL1_Image/             # Bài tập phân loại ảnh
│   ├── Image.html         # Trang báo cáo ảnh
│   ├── README.md          # Chi tiết kỹ thuật
│   ├── src/               # Mã nguồn Python
│   └── notebooks/         # Jupyter Notebooks
├── BTL1_Text/              # Bài tập phân loại văn bản
│   ├── Text.html          # Trang báo cáo văn bản
│   ├── README.md          # Chi tiết kỹ thuật
│   ├── src/               # Mã nguồn Python
│   └── notebooks/         # Jupyter Notebooks
└── BTL1_ImageText/        # Bài tập phân loại đa phương thức
    ├── ImageText.html     # Trang báo cáo multimodal
    └── Notebook.ipynb     # Jupyter Notebook
```

## Kết quả nổi bật

### Phân loại Ảnh
- **ResNet18**: ~95% accuracy
- **ViT-B/16**: ~93% accuracy
- **MobileNetV3**: ~91% accuracy (nhanh nhất)

### Phân loại Văn bản
- **BERT Full**: ~88% accuracy
- **BERT Freeze**: ~82% accuracy
- **Bi-LSTM**: ~85% accuracy

### Phân loại Đa phương thức
- **Few-shot (5-shot)**: ~78% accuracy
- **Zero-shot**: ~65% accuracy

## Các mở rộng đã thực hiện

1. **Interpretability**: Grad-CAM, Attention Visualization, LIME
2. **Error Analysis**: Confusion matrix, misclassified samples
3. **Fine-tuning Strategies**: Freeze vs Full vs Layer-wise LR
4. **Augmentation & Robustness**: RandAugment, MixUp
5. **Efficiency**: Model size, inference time, quantization
6. **Ensemble**: CNN + ViT, RNN + Transformer
7. **Calibration**: ECE, reliability diagram

## Công nghệ sử dụng

- **Framework**: PyTorch, Hugging Face Transformers
- **Visualization**: Chart.js, Matplotlib, Seaborn
- **Deployment**: GitHub Pages (HTML tĩnh)
- **IDE**: VS Code, Jupyter Notebook

## Liên kết

- **GitHub Repository**: [https://github.com/SchwiDolavn/CO3133--BB01](https://github.com/SchwiDolavn/CO3133--BB01)
- **Trang Landing Page**: [index.html](index.html)

---

*© 2026 - Nhóm BB01 - CO3133 Deep Learning*