import torch
import time
import os

def get_model_size(model):
    """Tính toán kích thước mô hình (MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time_pytorch(model, dataloader, device="cpu"):
    """Đo thời gian suy luận chuẩn xác trên PyTorch"""
    model.to(device)
    model.eval()
    
    # Warm up (Chạy nháp để khởi động phần cứng)
    dummy_input = next(iter(dataloader))
    input_ids = dummy_input['input_ids'].to(device)
    attention_mask = dummy_input['attention_mask'].to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask=attention_mask)
            
    # Đo thời gian thực tế
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
    """Nén mô hình bằng phương pháp Dynamic Quantization (Giảm kích thước, chuẩn PyTorch mới)"""
    print("Đang thực hiện Quantization (INT8)...")
    
    # CHUẨN MỚI: Sử dụng torch.ao.quantization thay vì torch.quantization
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, # Chỉ nén các lớp Linear
        dtype=torch.qint8  # Ép kiểu dữ liệu sang số nguyên 8-bit
    )
    return quantized_model