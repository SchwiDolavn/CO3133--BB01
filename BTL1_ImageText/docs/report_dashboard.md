# ROCO Radiology Multimodal Classification Results

Results dashboard for zero-shot and few-shot classification on ROCO Radiology. All models are compared with the same classification metrics: accuracy, precision, recall, F1, balanced accuracy, and confusion matrix.

- **Best zero-shot F1 macro:** 91.39% (CLIP-ViT-B/32)
- **Best zero-shot accuracy:** 93.59% (CLIP-ViT-B/32)
- **Best few-shot F1 macro:** 92.85% (OpenCLIP-ResNet50, 32-shot)
- **Best few-shot accuracy:** 93.88% (OpenCLIP-ResNet50, 32-shot)
- **Eval set:** 4,818 samples
- **Classes after keyword filter:** 5 (ct, xray_radiograph, mri, ultrasound, angiography)
- **Trainable params:** 1,285 to 1,285

## 4.3 Với tập dữ liệu đa phương thức
- **Zero-shot classification:** Không train classifier. Cả OpenCLIP ResNet50 và CLIP ViT-B/32 đều dùng image/caption embeddings so với class-name prompts.
- **Few-shot classification:** Train LogisticRegression nhẹ trên frozen multimodal features với `8` mẫu mỗi lớp.
- Input multimodal: image + caption/report text.

## 4.4 Metric đánh giá
- Metric chính: **accuracy** và **F1 macro**.
- Vì class distribution có thể mất cân bằng, notebook báo cáo thêm **F1 weighted**, **precision macro/weighted**, **recall macro/weighted**, và **balanced accuracy**.
- Mỗi cấu hình đều xuất **confusion matrix** và **classification report** theo từng lớp.

## Dataset processing
- Source: `/kaggle/input/datasets/shareef0612/roco-radiology/ROCO Radiology`
- Keyword filter enabled: `True`
- Target keyword groups: `{"ct": ["ct", "tomography", "computed tomography"], "xray_radiograph": ["xray", "x-ray", "radiograph", "radiographic", "anteroposterior"], "mri": ["mri", "magnetic", "resonance", "magnetic resonance"], "ultrasound": ["ultrasound", "sonograph", "sonographic", "sonography"], "angiography": ["angiography", "angiogram", "arteriography"]}`
- Keyword filtering is applied separately to train/val/test before sampling, feature extraction, and model training.
- Images: `/kaggle/input/datasets/shareef0612/roco-radiology/ROCO Radiology/images`
- CSV files: `train_data.csv`, `val_data.csv`, `test_data.csv`
- Label source: `keyword_filter_group: keyword_label`
- Keyword filter summary saved to `outputs/keyword_filter_summary.csv` and `outputs/keyword_class_distribution.csv`.
- Top classes kept: `5` max, min train per class `8`
- Splits: train/val/test kept separate; few-shot support samples only from train.

## Model configuration
- CLIP ViT-B/32 Transformers model: `openai/clip-vit-base-patch32`
- OpenCLIP ResNet50 model: `hf-hub:timm/resnet50_clip.openai`
- Image size: 224x224
- Few-shot classifier: LogisticRegression, class_weight=`balanced`
- k-shots: 8 per class

## Model comparison

| model             |   shots | mode                     |   accuracy |   balanced_accuracy |   precision_macro |   recall_macro |   f1_macro |   precision_weighted |   recall_weighted |   f1_weighted |   time_per_sample_ms |   trainable_params |
|:------------------|--------:|:-------------------------|-----------:|--------------------:|------------------:|---------------:|-----------:|---------------------:|------------------:|--------------:|---------------------:|-------------------:|
| CLIP-ViT-B/32     |       0 | Zero-shot classification |   0.935866 |            0.948937 |          0.894011 |       0.948937 |   0.913886 |             0.944176 |          0.935866 |      0.937277 |          2.63755e-05 |                  0 |
| OpenCLIP-ResNet50 |       0 | Zero-shot classification |   0.928601 |            0.95629  |          0.873839 |       0.95629  |   0.904253 |             0.942599 |          0.928601 |      0.931343 |          5.96294e-05 |                  0 |
| CLIP-ViT-B/32     |       8 | Few-shot classification  |   0.733292 |            0.725741 |          0.740819 |       0.725741 |   0.725481 |             0.746705 |          0.733292 |      0.72984  |          0.00241853  |               1285 |
| OpenCLIP-ResNet50 |       8 | Few-shot classification  |   0.771897 |            0.778606 |          0.773012 |       0.778606 |   0.766115 |             0.786119 |          0.771897 |      0.769114 |          0.00242932  |               1285 |
| CLIP-ViT-B/32     |      32 | Few-shot classification  |   0.921959 |            0.941255 |          0.899929 |       0.941255 |   0.918548 |             0.924759 |          0.921959 |      0.922389 |          0.00234242  |               1285 |
| OpenCLIP-ResNet50 |      32 | Few-shot classification  |   0.938771 |            0.951601 |          0.908618 |       0.951601 |   0.928466 |             0.941598 |          0.938771 |      0.93915  |          0.00241007  |               1285 |

## Release artifacts
- `outputs/model_comparison.csv`
- `outputs/keyword_filter_summary.csv`
- `outputs/keyword_class_distribution.csv`
- `outputs/classification_protocol.csv`
- `outputs/predictions_*.csv`
- `outputs/classification_report_*.csv`
- `outputs/confusion_matrix_*.csv`
- `models/classifier_*_8shot.pkl`
- `docs/plot/*.png` and `docs/plot/*.json`
- `streamlit_app.py`
