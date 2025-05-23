'''
demo for camera with OpenCV Haar Cascades using TFLite model
'''

import numpy as np
import cv2
import tensorflow as tf           # for tf.lite.Interpreter
from utils import putText, preprocess_input

# ------------------------------------------------------------------
# 1. 載入 TFLite 模型
# ------------------------------------------------------------------
tflite_model_path = r'face_weights\face_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# 取得輸入和輸出張量的 details
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ------------------------------------------------------------------
# 2. 標籤設定
# ------------------------------------------------------------------
gender_labels = ['Male', 'Female']
race_labels   = ['Whites', 'Blacks', 'Asian', 'Indian', 'Others']
age_labels    = np.arange(1, 94)

# ------------------------------------------------------------------
# 3. 初始化 Haar Cascade（人臉偵測）
# ------------------------------------------------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ------------------------------------------------------------------
# 4. 開啟攝影機
# ------------------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 轉灰階進行人臉偵測
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # 裁切並 resize
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        # 預處理（同訓練時）
        inp = preprocess_input(np.expand_dims(face_img.astype(np.float32), axis=0))

        # 設定輸入張量
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()

        # 取得三組輸出
        preds_ages    = interpreter.get_tensor(output_details[0]['index'])[0]
        preds_genders = interpreter.get_tensor(output_details[1]['index'])[0]
        preds_races   = interpreter.get_tensor(output_details[2]['index'])[0]

        # 找最大機率索引
        age_idx  = np.argmax(preds_ages)
        gen_idx  = np.argmax(preds_genders)
        race_idx = np.argmax(preds_races)

        # 畫人臉框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 疊加文字
        frame = putText(frame, f'Age: {age_labels[age_idx]}',       (255, 0, 0), (x, y - 20), size=20)
        frame = putText(frame, f'Gender: {gender_labels[gen_idx]}', (255, 0, 0), (x, y - 40), size=20)
        # frame = putText(frame, f'Race: {race_labels[race_idx]}',    (255, 0, 0), (x, y - 60), size=20)

    # 顯示結果
    cv2.imshow('TFLite Haar Cascade Face Demo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理
cap.release()
cv2.destroyAllWindows()
