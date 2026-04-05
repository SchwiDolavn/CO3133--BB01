import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_class_weights(y_train):
    """
    Tính toán trọng số lớp học tự động từ dữ liệu.
    Dùng tham số class_weight trong model.fit(..., class_weight=weights).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return dict(zip(classes, weights))

def focal_loss_custom(gamma=2.0, alpha=0.25):
    """
    Hàm Focal Loss dành cho bài toán phân loại nhiều lớp (Multi-class).
    Phạt nặng các dự đoán sai mà mô hình tự tin, giúp ích cho Imbalanced Data.
    """
    def focal_loss_fn(y_true, y_pred):
        # Đảm bảo y_true và y_pred đúng kiểu dữ liệu
        y_true = tf.cast(y_true, tf.int32)
        
        # Lấy xác suất của lớp đúng
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Chuyển y_true thành one-hot vector
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        
        # Tính Focal Loss
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.math.pow((1 - y_pred), gamma)
        
        loss = weight * cross_entropy
        return tf.reduce_sum(loss, axis=1)
        
    return focal_loss_fn