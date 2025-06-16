import numpy as np
import cv2
import tensorflow as tf           # for Keras model and gradients
from utils import putText
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess_input
import os

SCALE = 4

# 全域預設 backbone，可設為 'vggface' 或 'resnet'
BACKBONE_TYPE = 'vggface'  # 如使用 ResNet backbone，可改成 'resnet'

# ------------------------------------------------------------------
# 新增：根據 BACKBONE_TYPE 選擇預處理函式
# ------------------------------------------------------------------
def preprocess_backbone(x: np.ndarray) -> np.ndarray:
    """
    根據全域 BACKBONE_TYPE，對輸入影像 x 進行相應的 preprocess_input 處理。
    """
    if BACKBONE_TYPE.lower().startswith('resnet'):
        return resnet_preprocess_input(x)
    else:
        return vggface_preprocess_input(x)


def vggface_preprocess_input(x, data_format=None, version=1):
    x_temp = np.copy(x)
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863

    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError

    return x_temp


# ------------------------------------------------------------------
# 0. Grad-CAM: 產生熱力圖
# ------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=0):
    # 建立 grad model，輸出最後 conv 層特徵圖與目標輸出
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output[pred_index]]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# ------------------------------------------------------------------
# 1. 載入 Keras 模型
# ------------------------------------------------------------------
def load_keras_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# ------------------------------------------------------------------
# 2. 初始化 Haar Cascade（人臉偵測器）
# ------------------------------------------------------------------
def init_face_detector():
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(cascade_path)

# ------------------------------------------------------------------
# 3. 處理單張人臉 Region，做推論 + Grad-CAM
# ------------------------------------------------------------------
def predict_and_cam(model, last_conv_layer, face_img):
    # 選擇預處理：根據 BACKBONE_TYPE 動態使用 VGGFace 或 ResNet 的 preprocess_input
    x = face_img.astype(np.float32)
    x = preprocess_backbone(x)
    inp = np.expand_dims(x, axis=0)
    # 取得預測（直接呼叫 model，避免 DataAdapter 匯入 pandas）
    outputs = model(inp, training=False)
    # 若模型回傳多個輸出，取前兩作為 gender 和 age
    preds_gender, preds_age = outputs[0], outputs[1]
    # 產生 Grad-CAM 熱力圖 (性別分類, 索引 0)
    heatmap = make_gradcam_heatmap(inp, model, last_conv_layer, pred_index=0)
    return float(preds_age[0][0]), preds_gender[0][0], heatmap

# ------------------------------------------------------------------
# 4. 繪製並疊加 Grad-CAM
# ------------------------------------------------------------------
def overlay_heatmap(frame, heatmap, x, y, w, h, alpha=0.4):
    heatmap = cv2.resize(heatmap, (w, h))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame[y:y+h, x:x+w], 1 - alpha, heatmap, alpha, 0)
    frame[y:y+h, x:x+w] = overlay
    return frame

# ------------------------------------------------------------------
# 5. 繪製人臉框與屬性文字
# ------------------------------------------------------------------
def annotate_frame(frame, faces, preds_list, labels):
    gender_labels = ['Male', 'Female']

    for (x, y, w, h), (age_val, gender_prob, heatmap) in zip(faces, preds_list):
        # # CAM 疊加
        # frame = overlay_heatmap(frame, heatmap, x, y, w, h)
        # 文字
        gen_idx = 1 if gender_prob > 0.5 else 0
        age_text = f'Age: {age_val:.1f}'
        gender_text = f'Gender: {gender_labels[gen_idx]} ({gender_prob:.2f})'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame = putText(frame, age_text,    (0, 255, 0), (x, y - 30*SCALE), size=30*SCALE)
        frame = putText(frame, gender_text, (0, 255, 0), (x, y - 30*SCALE*2), size=30*SCALE)
    return frame

# ------------------------------------------------------------------
# 6. 處理影格：偵測 → 裁切 → 推論+CAM → 標註
# ------------------------------------------------------------------
def process_frame(frame, face_cascade, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
    preds_list = []
    for (x, y, w, h) in faces:
        # 裁切並 resize，保留三通道彩色輸入 (128,128,3)
        face = cv2.resize(frame[y:y+h, x:x+w], (128, 128))
        # 如模型在 RGB 空間訓練，可加這行進行 BGR→RGB 轉換
        face_img = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 動態決定最後 conv 層名稱：VGGFace 常為 'conv5_3'，ResNet50 常為 'conv5_block3_out'
        if BACKBONE_TYPE.lower().startswith('resnet'):
            last_conv = 'conv5_block3_out'
        else:
            last_conv = 'conv5_3'
        age, gender_prob, heatmap = predict_and_cam(model, last_conv, face_img)
        preds_list.append((age, gender_prob, heatmap))
    if faces is not None and len(faces) > 0:
        frame = annotate_frame(frame, faces, preds_list, None)
    return frame

# ------------------------------------------------------------------
# 7. 即時影像處理
# ------------------------------------------------------------------
def process_realtime(face_cascade, model):
    # 即時模式時暫時將 SCALE 設為 1
    global SCALE
    old_SCALE = SCALE
    SCALE = 1
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機。")
        return
    print("按 'q' 結束。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = process_frame(frame, face_cascade, model)
        cv2.imshow('Grad-CAM Gender/Age Demo', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    # 還原 SCALE 為靜態圖片模式預設值
    SCALE = old_SCALE


# ------------------------------------------------------------------
# 8. 靜態圖片辨識
# ------------------------------------------------------------------
def process_image(image_path, face_cascade, model, save_path=None):
    """對單張圖片做人臉偵測、推論 + Grad-CAM，並顯示或儲存結果。"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片：{image_path}")
        return
    out = process_frame(img, face_cascade, model)
    if save_path:
        cv2.imwrite(save_path, out)
        print(f"結果已儲存至 {save_path}")
    else:
        # 縮小視窗為原尺寸 1/scale
        h, w = out.shape[:2]
        small = cv2.resize(out, (w // SCALE, h // SCALE))
        cv2.imshow('Static Image Inference', small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ------------------------------------------------------------------
# 9. 主程式
# ------------------------------------------------------------------
def main():
    # 初始化偵測器
    face_cascade = init_face_detector()
    # 靜態圖片路徑
    img_path = r'tmp\archive\train\21-30\8.jpg'
    os.makedirs('results', exist_ok=True)
    # 定義多個模型與對應 backbone
    model_configs = [
        (r'model\best_model_vgg.h5', 'vggface'),
        # (r'model\best_model_resnet.h5', 'resnet'),
    ]
    for model_path, backbone in model_configs:
        # 設定全域 BACKBONE_TYPE
        global BACKBONE_TYPE
        BACKBONE_TYPE = backbone
        # 載入模型
        model = load_keras_model(model_path)
        # process_image(img_path, face_cascade, model)
        # 啟動即時辨識
        process_realtime(face_cascade, model)

if __name__ == '__main__':
    main()
