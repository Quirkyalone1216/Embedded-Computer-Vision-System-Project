import numpy as np
import cv2
import tensorflow as tf           # for tf.lite.Interpreter
from utils import putText, preprocess_input
import os

# ------------------------------------------------------------------
# 1. 載入 TFLite 模型並取得張量資訊
# ------------------------------------------------------------------
def load_tflite_model(model_path):
    """
    載入 TFLite 模型並分配張量緩衝區。
    回傳 interpreter、input_details、output_details 三個物件。
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details


# ------------------------------------------------------------------
# 2. 取得標籤清單（Labels）
# ------------------------------------------------------------------
def get_labels():
    """
    回傳三組標籤：性別、種族、年齡。
    年齡以 1~93 為範圍。
    """
    gender_labels = ['Male', 'Female']
    race_labels   = ['Whites', 'Blacks', 'Asian', 'Indian', 'Others']
    age_labels    = np.arange(94)     # 0 至 93，共 94 種年齡
    return gender_labels, race_labels, age_labels


# ------------------------------------------------------------------
# 3. 初始化 Haar Cascade（人臉偵測器）
# ------------------------------------------------------------------
def init_face_detector():
    """
    使用 OpenCV 內建 Haar Cascade XML 檔，初始化人臉偵測器。
    回傳一個 cv2.CascadeClassifier 物件。
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade


# ------------------------------------------------------------------
# 4. 針對單張人臉 Region 進行前處理並做推論
# ------------------------------------------------------------------
def predict_face_attributes(interpreter, input_details, output_details, face_img):
    """
    輸入：
      - interpreter: 已配置好的 TFLite Interpreter
      - input_details, output_details: 取得的張量資訊
      - face_img: 已經裁成 (H, W, 3) BGR 影像（此範例假設大小為 200×200）
    會將 face_img 做 preprocess_input，送入 interpreter，invoke 之後回傳三組機率向量。
    回傳：
      - preds_ages: 長度 93 的年齡機率向量
      - preds_genders: 長度 2 的性別機率向量
      - preds_races: 長度 5 的種族機率向量
    """
    # 1. 確保輸入是 float32，並加上 batch 維度
    inp = preprocess_input(np.expand_dims(face_img.astype(np.float32), axis=0))
    
    # 2. 設定輸入張量
    interpreter.set_tensor(input_details[0]['index'], inp)
    # 3. 執行推論
    interpreter.invoke()
    
    # 4. 取得三組輸出
    preds_ages    = interpreter.get_tensor(output_details[0]['index'])[0]
    preds_genders = interpreter.get_tensor(output_details[1]['index'])[0]
    preds_races   = interpreter.get_tensor(output_details[2]['index'])[0]
    return preds_ages, preds_genders, preds_races


# ------------------------------------------------------------------
# 5. 在原影格上繪製人臉框與屬性文字
# ------------------------------------------------------------------
def annotate_frame(frame, faces, preds_list, labels_list):
    """
    將每個人臉 ROI 的預測結果，繪製到原始影格 frame 上。
    參數：
      - frame: 原始 BGR 影像
      - faces: Haar Cascade 回傳的 list of (x, y, w, h)
      - preds_list: 每個人臉對應的 (preds_ages, preds_genders, preds_races) tuple list
      - labels_list: 三組標籤 (gender_labels, race_labels, age_labels)
    回傳：
      - 修改後的 frame（已經畫好矩形框與文字）
    """
    gender_labels, race_labels, age_labels = labels_list
    
    for (x, y, w, h), (preds_ages, preds_genders, preds_races) in zip(faces, preds_list):
        # 計算年齡分類的期望值 (加權平均)
        age_val    = preds_ages.dot(age_labels)
        gen_idx    = np.argmax(preds_genders)
        race_idx   = np.argmax(preds_races)
        
        age_text   = f'Age: {age_val:.1f}'
        gender_text= f'Gender: {gender_labels[gen_idx]}'
        race_text  = f'Race: {race_labels[race_idx]}'  # 若要顯示種族，可取消註解
        
        # 繪製人臉框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 疊加文字：年齡、性別（種族留作範例，若需顯示可自行取消）
        frame = putText(frame, age_text,    (255, 0, 0), (x, y - 20), size=20)
        frame = putText(frame, gender_text, (255, 0, 0), (x, y - 40), size=20)
        frame = putText(frame, race_text,   (255, 0, 0), (x, y - 60), size=20)
    
    return frame


# ------------------------------------------------------------------
# 6. 處理單張影格：偵測臉部 → 裁切 → 推論 → 標註
# ------------------------------------------------------------------
def process_frame(frame, face_cascade, interpreter, input_details, output_details, labels_list):
    """
    對單張 frame 執行以下步驟：
      1. 轉灰階並用 Haar Cascade 偵測人臉
      2. 裁切每個人臉為 200×200
      3. 呼叫 predict_face_attributes 做推論
      4. 呼叫 annotate_frame 在 frame 上繪製結果
    回傳修改後的 frame。
    """
    # 1. 轉灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 2. 多尺度偵測人臉，回傳 list of (x, y, w, h)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    preds_list = []
    # 3. 針對每張偵測到的人臉，裁切 → resize → 推論
    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        preds_ages, preds_genders, preds_races = predict_face_attributes(
            interpreter, input_details, output_details, face_img
        )
        preds_list.append((preds_ages, preds_genders, preds_races))
    
    # 4. 如有偵測到至少一張人臉，則在 frame 上繪製結果
    if len(faces) > 0:
        frame = annotate_frame(frame, faces, preds_list, labels_list)
    
    return frame


# ------------------------------------------------------------------
# 7a. 處理靜態圖片：讀取、推論與顯示
# ------------------------------------------------------------------
def process_static_image(img_path, face_cascade, interpreter, input_details, output_details, labels_list):
    """
    讀取指定路徑的靜態圖片，進行人臉偵測與屬性推論，並顯示標註後結果。
    """
    if not os.path.exists(img_path):
        print(f"Static image not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"無法讀取靜態圖片：{img_path}")
        return
    annotated_img = process_frame(
        img, face_cascade, interpreter, input_details, output_details, labels_list
    )
    # 縮小顯示視窗：將影像縮放至原始尺寸一半
    h, w = annotated_img.shape[:2]
    small_img = cv2.resize(annotated_img, (int(w * 0.5), int(h * 0.5)))
    cv2.imshow('Static Image Test', small_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Static Image Test')


# ------------------------------------------------------------------
# 7b. 處理即時影像：開啟攝影機並持續擷取／顯示影格
# ------------------------------------------------------------------
def process_realtime(face_cascade, interpreter, input_details, output_details, labels_list):
    """
    開啟預設攝影機，對每個影格執行人臉偵測與屬性推論，並即時顯示標註結果。按 'q' 鍵結束。
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法開啟攝影機，請檢查攝影機是否可用。")
        return
    print("按下 'q' 鍵即可結束程式。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("擷取影格失敗，結束程式。")
            break
        annotated_frame = process_frame(
            frame, face_cascade, interpreter, input_details, output_details, labels_list
        )
        cv2.imshow('TFLite Haar Cascade Face Demo', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------
# 7. 主程式：載入模型、初始化並呼叫靜態／即時流程
# ------------------------------------------------------------------
def main():
    # 7-1. 載入模型
    model_path = r'face_weights\face_model.tflite'
    interpreter, input_details, output_details = load_tflite_model(model_path)

    # 7-2. 取得標籤
    gender_labels, race_labels, age_labels = get_labels()
    labels_list = (gender_labels, race_labels, age_labels)

    # 7-3. 初始化人臉偵測器
    face_cascade = init_face_detector()

    # 7-4. 處理靜態圖片測試
    test_img_path = r'tmp\archive\test\18-20\26.jpg'
    process_static_image(test_img_path, face_cascade, interpreter, input_details, output_details, labels_list)

    # # 7-5. 處理即時影像
    # process_realtime(face_cascade, interpreter, input_details, output_details, labels_list)


if __name__ == "__main__":
    main()
