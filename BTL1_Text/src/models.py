from datasets import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import TextVectorization, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Sequential

def model(model_type, num_classes=0, ):
    if model_type == 'RNN_LSTM':
        model_LSTM = Sequential([tf.keras.Input(shape=(1,), dtype=tf.string), text_vectorizer, Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, mask_zero=True), Bidirectional(LSTM(32)), Dense(5, activation='softmax')])

        model_LSTM.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

        model_LSTM.summary()
    elif model_type == 'Transformer':
        pass
    elif model_type == '':
        pass

class EnsembleModel(nn.Module):
    """Kết hợp dự đoán của 2 mô hình bằng cách lấy trung bình cộng logits"""
    def __init__(self, modelA, modelB):
        super(EnsembleModel, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        
    def forward(self, x):
        outA = self.modelA(x)
        outB = self.modelB(x)
        return (outA + outB) / 2.0