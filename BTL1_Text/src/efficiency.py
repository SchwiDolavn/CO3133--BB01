import torch
import time
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_model_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)
    return size_mb

def measure_inference_time_pytorch(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    
    dummy_input = next(iter(dataloader))
    input_ids = dummy_input['input_ids'].to(device)
    attention_mask = dummy_input['attention_mask'].to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask=attention_mask)
            
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            _ = model(ids, attention_mask=mask)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / len(dataloader.dataset)
    return avg_time

def apply_quantization(model):    
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8  
    )
    return quantized_model

def plot_efficiency_tradeoffs(results_list):
    if not results_list:
        return
        
    df = pd.DataFrame(results_list)
    
    df['Accuracy_Num'] = df['Accuracy'].str.rstrip('%').astype('float')
    df['Inference Time (s)'] = df['Inference Time (s)'].astype('float')
    df['Model Size (MB)'] = df['Model Size (MB)'].astype('float')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.scatterplot(data=df, x='Inference Time (s)', y='Accuracy_Num', hue='Scenario', s=200, ax=ax1, palette='tab10')
    ax1.set_title('Trade-off: Tốc độ suy luận vs Độ chính xác')
    ax1.set_xlabel('Thời gian suy luận 1 mẫu (giây) -> Càng nhỏ càng tốt')
    ax1.set_ylabel('Độ chính xác (Accuracy %)')
    ax1.grid(True, linestyle='--', alpha=0.7)

    sns.scatterplot(data=df, x='Model Size (MB)', y='Accuracy_Num', hue='Scenario', s=200, ax=ax2, palette='tab10', marker='X')
    ax2.set_title('Trade-off: Kích thước mô hình vs Độ chính xác')
    ax2.set_xlabel('Kích thước mô hình (MB) -> Càng nhỏ càng tốt')
    ax2.set_ylabel('Độ chính xác (Accuracy %)')
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.get_legend().remove()

    plt.tight_layout()
    plt.savefig('../result/Efficiency_Tradeoffs.png', bbox_inches='tight')
    plt.close()
