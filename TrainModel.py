import os
import random
import json
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight as cw

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

from keras_vggface.vggface import VGGFace  
from keras_vggface.utils import preprocess_input  

warnings.filterwarnings('ignore')
gender_dict = {0: 'Male', 1: 'Female'}


def load_metadata(base_dir):
    """讀取 UTKFace 資料，回傳 DataFrame 包含路徑、年齡、性別、年齡桶。"""
    paths, ages, genders = [], [], []
    for fn in tqdm(os.listdir(base_dir), desc="Loading metadata"):
        full_path = os.path.join(base_dir, fn)
        age, gender = map(int, fn.split('_')[:2])
        paths.append(full_path)
        ages.append(age)
        genders.append(gender)
    df = pd.DataFrame({
        'image': paths,
        'age': ages,
        'gender': genders
    })
    # 年齡分桶（0-9,10-19,…,90+），大於 99 的歸到第 9 桶
    df['age_bin'] = (df['age'] // 10).clip(upper=9)
    return df


def extract_features(image_paths, target_size=(128, 128)):
    """將影像讀為 RGB 並 resize，再正規化至 [0,1]。"""
    feats = []
    for path in tqdm(image_paths, desc="Extracting features"):
        img = load_img(path, color_mode='rgb', target_size=target_size)
        x = np.array(img, dtype=np.float32)
        # 使用 VGGFace 專用預處理，對輸入做 mean subtraction 等處理 :contentReference[oaicite:1]{index=1}
        x = preprocess_input(x)  
        feats.append(x)
    X = np.stack(feats, axis=0)
    # 移除 /255，因 preprocess_input 已處理輸入分佈，避免額外縮放導致輸入分布失真
    return X


def preprocess_data(base_dir):
    """整合讀檔與特徵萃取，並回傳所有標籤：gender, age, age_class。"""
    df = load_metadata(base_dir)
    X = extract_features(df['image'])
    y_gender = df['gender'].values
    y_age = df['age'].values
    # one-hot 之前先確保 age_bin 在 0~9 範圍內
    bins = df['age_bin'].clip(0, 9).astype(int)
    y_age_class = tf.keras.utils.to_categorical(bins, num_classes=10)
    return X, y_gender, y_age, y_age_class


def oversample_age_buckets(X, y_gender, y_age):
    """依年齡段 (0-5,5-20,20-60,60+) 做平衡過採樣。"""
    buckets = {
        '0_5':    np.where(y_age < 5)[0],
        '5_20':   np.where((y_age >= 5) & (y_age < 20))[0],
        '20_60':  np.where((y_age >= 20) & (y_age < 60))[0],
        '60_plus':np.where(y_age >= 60)[0],
    }
    max_size = max(len(idxs) for idxs in buckets.values() if len(idxs) > 0)
    new_idxs = []
    for idxs in buckets.values():
        if len(idxs) == 0:
            continue
        choices = np.random.choice(idxs, size=max_size, replace=True)
        new_idxs.extend(choices.tolist())
    new_idxs = np.array(new_idxs)
    return X[new_idxs], y_gender[new_idxs], y_age[new_idxs]


def compute_sample_weights(y_gender):
    """依性別標籤計算 balanced sample weights。"""
    weights = cw.compute_class_weight('balanced', classes=np.unique(y_gender), y=y_gender)
    cw_dict = dict(enumerate(weights))
    return np.vectorize(cw_dict.get)(y_gender)


def build_model(input_shape=(128,128,3), l2_reg=1e-4, dropout_rate=0.5):
    """建立多輸出 (gender, age_reg, age_class) CNN。"""
    # 使用 VGGFace (VGG16) 做 backbone，去除原頂層，並採用全域平均池化取得特徵
    base_model = VGGFace(include_top=False,
                         input_shape=input_shape,
                         pooling='avg')  
    # 凍結 backbone 層，加速微調並保留預訓練特徵
    for layer in base_model.layers:
        layer.trainable = False  
    x = base_model.output

    # 性別分支
    g = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    g = Dropout(dropout_rate)(g)
    out_gender = Dense(1, activation='sigmoid', name='gender_out', kernel_regularizer=l2(l2_reg))(g)

    # 年齡回歸分支：改用線性輸出，以便直接預測整體年齡
    a = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    a = Dropout(dropout_rate)(a)
    out_age_reg = Dense(1, activation='linear', name='age_out_reg', kernel_regularizer=l2(l2_reg))(a)

    # 年齡分類分支
    c = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    c = Dropout(dropout_rate)(c)
    out_age_class = Dense(10, activation='softmax', name='age_out_class', kernel_regularizer=l2(l2_reg))(c)

    # 以 VGGFace 的輸入作為模型輸入，三支分支結構不變
    model = Model(inputs=base_model.input,
                  outputs=[out_gender, out_age_reg, out_age_class])
    # 調小學習率以穩定回歸分支訓練
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss={
            'gender_out':    'binary_crossentropy',
            'age_out_reg':   'mae',
            'age_out_class': 'categorical_crossentropy'
        },
        loss_weights={
            'gender_out':    1.0,
            'age_out_reg':   1.0,
            'age_out_class': 1.0
        },
        metrics={
            'gender_out':    ['accuracy'],
            'age_out_reg':   ['mae'],
            'age_out_class': ['accuracy']
        },
        weighted_metrics=[]
    )
    return model


def train_model(model, X, y_gender, y_age, y_age_class,
                sample_weights=None,
                checkpoint_path='results/best_model.h5',
                epochs=50, batch_size=256, val_split=0.2):
    """訓練模型並儲存最佳權重。"""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    # 新增 EarlyStopping 與 ReduceLROnPlateau 以監控 age_out_reg 的驗證 MAE
    ckpt = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_age_out_reg_mae', patience=5, restore_best_weights=True, verbose=1)
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_age_out_reg_mae', factor=0.5, patience=3, verbose=1)
    fit_args = {
        'x': X,
        'y': {
            'gender_out':    y_gender,
            'age_out_reg':   y_age,
            'age_out_class': y_age_class
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'validation_split': val_split,
        'callbacks': [ckpt, es, rlrop]
    }
    if sample_weights is not None:
        fit_args['sample_weight'] = {'gender_out': sample_weights}
    # 先 shuffle 資料，避免 validation_split 切到偏序列
    idxs = np.arange(len(X))
    np.random.shuffle(idxs)
    # 重排 X 與標籤
    X_shuf = X[idxs]
    yg = fit_args['y']['gender_out'][idxs]
    ya = fit_args['y']['age_out_reg'][idxs]
    yc = fit_args['y']['age_out_class'][idxs]
    # 若有 sample_weights，需同步重排
    if 'sample_weight' in fit_args:
        sw = fit_args['sample_weight']['gender_out'][idxs]
        fit_args['sample_weight'] = {'gender_out': sw}
    # 更新 fit_args
    fit_args['x'] = X_shuf
    fit_args['y'] = {'gender_out': yg, 'age_out_reg': ya, 'age_out_class': yc}
    return model.fit(**fit_args)


def evaluate_model(model, X, y_gender, y_age, batch_size=128, out_dir='results'):
    """在 CPU 上批次推論並繪製混淆矩陣與年齡誤差圖。"""
    os.makedirs(out_dir, exist_ok=True)
    steps = math.ceil(len(X) / batch_size)
    genders, ages = [], []
    with tf.device('/CPU:0'):
        for i in range(steps):
            start = i * batch_size
            end = min((i+1) * batch_size, len(X))
            preds = model.predict(X[start:end], batch_size=batch_size)
            genders.append(preds[0])
            ages.append(preds[1])
    y_pred_gender = np.concatenate(genders, axis=0).round().astype(int).flatten()
    y_pred_age    = np.concatenate(ages, axis=0).flatten()
    # 性別混淆矩陣
    cm = confusion_matrix(y_gender, y_pred_gender)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Male','Female'])
    disp.plot(cmap='Blues')
    plt.title('Gender Confusion Matrix')
    plt.savefig(os.path.join(out_dir, 'gender_confusion_matrix.png'))
    plt.close()
    # 年齡誤差直方圖（以百分比表示）
    errors = np.abs(y_pred_age - y_age)
    plt.figure()
    # 計算每個樣本的權重，使直方圖顯示百分比
    weights = np.ones_like(errors) / len(errors) * 100
    plt.hist(errors, bins=50, weights=weights, edgecolor='black')
    plt.title('Age Error Distribution (%)')
    plt.xlabel('Absolute Error (years)')
    plt.ylabel('Percentage (%)')
    plt.savefig(os.path.join(out_dir, 'age_error_histogram.png'))
    plt.close()


def plot_metrics(history, out_dir='results'):
    """繪製訓練曲線：Accuracy、Loss、Age MAE。"""
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(len(history.history['gender_out_accuracy']))

    plt.figure()
    plt.plot(epochs, history.history['gender_out_accuracy'], label='Train Acc')
    plt.plot(epochs, history.history['val_gender_out_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy_graph.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history.history['gender_out_loss'], label='Train Loss')
    plt.plot(epochs, history.history['val_gender_out_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_graph.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history.history['age_out_reg_mae'], label='Train Age MAE')
    plt.plot(epochs, history.history['val_age_out_reg_mae'], label='Val Age MAE')
    plt.title('Age MAE')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'age_mae_graph.png'))
    plt.close()


def generate_predictions(model, X, y_gender, y_age,
                         n_samples=100, out_path='results/predictions.json'):
    """隨機抽樣預測並輸出 JSON。"""
    idxs = random.sample(range(len(X)), n_samples)
    preds = model.predict(X[idxs])
    results = []
    for i, idx in enumerate(idxs):
        pg = int(round(preds[0][i][0]))
        # 直接使用回歸輸出作為年齡預測
        pa = int(round(preds[1][i][0]))
        results.append({
            'index': idx,
            'true': {'gender': gender_dict[y_gender[idx]], 'age': int(y_age[idx])},
            'pred': {'gender': gender_dict[pg], 'age': pa}
        })
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    base_dir = r'dataset\archive\UTKFace'
    X, y_gender, y_age, y_age_class = preprocess_data(base_dir)

    X_bal, y_gender_bal, y_age_bal = oversample_age_buckets(X, y_gender, y_age)
    # 重新計算 oversample 後的 age_bin，並夾至 0~9
    y_age_bin_bal   = np.clip(y_age_bal // 10, 0, 9)
    y_age_class_bal = tf.keras.utils.to_categorical(y_age_bin_bal, num_classes=10)

    sample_weights = compute_sample_weights(y_gender_bal)

    model = build_model(input_shape=X_bal.shape[1:])
    history = train_model(
        model,
        X_bal, y_gender_bal, y_age_bal, y_age_class_bal,
        sample_weights=sample_weights
    )

    plot_metrics(history)
    evaluate_model(model, X, y_gender, y_age)
    generate_predictions(model, X, y_gender, y_age)


if __name__ == '__main__':
    main()
