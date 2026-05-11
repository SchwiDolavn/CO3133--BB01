import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import os

def plot_training_history(log_history, scenario_name):
    train_epochs = [entry['epoch'] for entry in log_history if 'loss' in entry]
    train_loss = [entry['loss'] for entry in log_history if 'loss' in entry]
    val_epochs = [entry['epoch'] for entry in log_history if 'eval_loss' in entry]
    val_loss = [entry['eval_loss'] for entry in log_history if 'eval_loss' in entry]
    
    val_acc_epochs = [entry['epoch'] for entry in log_history if 'eval_accuracy' in entry]
    val_acc = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]
    val_f1 = [entry.get('eval_f1', entry.get('eval_f1_macro', None)) for entry in log_history if 'eval_loss' in entry]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    if train_loss: ax1.plot(train_epochs, train_loss, label='Train Loss', marker='o')
    if val_loss: ax1.plot(val_epochs, val_loss, label='Val Loss', marker='s')
    ax1.set_title('Biểu đồ Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    
    if val_acc:
        ax2.plot(val_acc_epochs, val_acc, label='Val Accuracy', color='green', marker='^')
    val_f1_clean = [f for f in val_f1 if f is not None]
    if val_f1_clean and len(val_f1_clean) == len(val_epochs):
        ax2.plot(val_epochs, val_f1_clean, label='Val F1-Score', color='orange', marker='d')
        
    ax2.set_title('Biểu đồ Metrics (Acc & F1)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'../result/{scenario_name}_training_history.png')
    plt.close()

def evaluate_model(y_true, y_pred, classes, scenario_name):    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(f'Confusion Matrix - {scenario_name}')
    
    plt.savefig(f'../result/{scenario_name}_confusion_matrix.png')
    plt.close()

def error_analysis(texts, y_true, y_pred_probs, label_encoder, scenario_name, top_n=5):
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    confidences = np.max(y_pred_probs, axis=1)
    errors_idx = np.where(y_pred_classes != y_true)[0]
    
    error_data = []
    for idx in errors_idx:
        py_idx = int(idx) 
        error_data.append({
            'Text': texts[py_idx],
            'True_Label': label_encoder.inverse_transform([y_true[py_idx]])[0],
            'Predicted_Label': label_encoder.inverse_transform([y_pred_classes[py_idx]])[0],
            'Confidence': confidences[py_idx]
        })
        
    df_errors = pd.DataFrame(error_data)
    df_errors = df_errors.sort_values(by='Confidence', ascending=False)
    
    file_path = f'../result/{scenario_name}_error_analysis.csv'
    df_errors.to_csv(file_path, index=False)

def export_results_to_csv(y_true, y_pred, classes, model_name):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    file_path = f"../result/{model_name}_classification_report.csv"
    df_report.to_csv(file_path)
    return df_report

def plot_comprehensive_comparison(results_list):
    """Vẽ biểu đồ so sánh Accuracy và F1 cho tất cả kịch bản"""
    df = pd.DataFrame(results_list)
    
    df['Accuracy_Num'] = df['Accuracy'].str.rstrip('%').astype('float')
    if 'F1-Score' in df.columns:
        df['F1_Num'] = df['F1-Score'].str.rstrip('%').astype('float')
    
    melted_df = pd.melt(df, id_vars=['Scenario'], 
                        value_vars=['Accuracy_Num', 'F1_Num'] if 'F1-Score' in df.columns else ['Accuracy_Num'],
                        var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(15, 7))
    sns.barplot(data=melted_df, x='Scenario', y='Score', hue='Metric', palette='Set2')
    plt.title('So sánh Accuracy và F1-Score giữa các mô hình')
    plt.ylim(0, 100) 
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Loại chỉ số')
    plt.tight_layout()
    plt.savefig('../result/Global_Comparison_Report.png')
    plt.close()