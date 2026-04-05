import time
import tensorflow as tf

def measure_inference_time(model, dataset, num_samples=1000):
    """Đo lường thời gian dự đoán của mô hình."""
    start_time = time.time()
    _ = model.predict(dataset.take(1)) # Warm up
    
    start_time = time.time()
    # Chạy inference thực tế
    _ = model.predict(dataset)
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"Tổng thời gian inference: {total_time:.4f} giây")
    return total_time