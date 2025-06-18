## 一、專案名稱

**Python 年齡與性別預測訓練與推論系統**

---

## 二、專案概述

本專案提供一個完整的從資料前處理、模型訓練、量化轉檔到即時推論的 Python Pipeline，透過已標註的 UTKFace 資料集，以 VGGFace 為 backbone 訓練多輸出神經網路，同時支援 TensorFlow Lite 轉檔及 OpenCV 即時推論介面。並提供 Android 專案，包含 Android Studio 專案、Android.mk 與 CMakeLists.txt，支援 Android NDK 建置。

---

## 三、主要功能

1. **資料前處理**：自動讀取 UTKFace 資料集、解析年齡與性別標籤，支援影像縮放、正規化與分桶（年齡分類）。
2. **模型訓練**：以 Keras 建構多輸出模型，分支同時輸出性別（二分類）、年齡回歸與年齡分組（十分類），整合早停與最佳模型權重保存。
3. **效能評估**：繪製訓練/驗證階段的 Loss、Accuracy、MAE 曲線，並生成混淆矩陣與年齡誤差直方圖。
4. **TensorFlow Lite 轉檔**：將訓練後的 `.h5` 模型轉為動態範圍量化的 `.tflite`，支援書寫到檔案並驗證轉檔結果。
5. **推論腳本**：利用 OpenCV Haar Cascade 偵測人臉，支援靜態影像或攝影機串流的即時推論，並可疊加預測結果與 Grad-CAM 熱力圖。
6. **Android 專案**：提供 Android 專案，包含 Android Studio 專案、Android.mk 與 CMakeLists.txt，支援 Android NDK 建置。

---

## 四、專案結構

```
├── TrainModel.py       # 訓練主程式：資料讀取、模型建置、訓練、評估與預測輸出
├── utils.py            # 工具函式：VGGFace 前處理、Label Decode、文字標註、TFLite 轉檔
├── Inference.py        # 推論主程式：人臉偵測、推論、Grad-CAM 整合與顯示/存檔
├── results/            # 訓練結果（模型權重、指標圖、Log）
├── data/               # 放置原始 UTKFace 影像資料集
└── README.md           # 專案說明
└── AndroidCode         # Android 專案
```

---

## 五、環境與安裝

1. 建議使用 Python 3.8 以上版本。
2. 安裝相依套件：

```bash
pip install -r requirements.txt
```

*requirements.txt* 範例內容：

```
tensorflow>=2.8  
opencv-python  
numpy  
pandas  
matplotlib  
tqdm  
Pillow  
```

---

## 六、使用說明

1. **資料前處理與模型訓練**
   ```bash
   python TrainModel.py \
     --data_dir ./data/UTKFace \
     --output_dir ./results \
     --epochs 50
   ```
2. **TFLite 轉檔**
   ```bash
   python -c "from utils import ConvertH5TOTflite; ConvertH5TOTflite('results/best_model.h5')"
   ```
3. **靜態影像推論**
   ```bash
   python Inference.py --image_path ./test.jpg --use_cam false
   ```
4. **即時攝影機推論**
   ```bash
   python Inference.py --use_cam true
   ```

---

## 七、未來擴充

- 支援更多人臉偵測演算法（MediaPipe、Dlib）
- 加入模型微調與超參數自動搜尋功能

---

## 八、參考資料

- VGGFace2 Pretrained Weights\
  [https://github.com/rcmalli/keras-vggface](https://github.com/rcmalli/keras-vggface)
- TensorFlow Lite Converter\
  [https://www.tensorflow.org/lite/convert](https://www.tensorflow.org/lite/convert)
- OpenCV Haar Cascade\
  [https://docs.opencv.org/4.x/db/d28/tutorial\_cascade\_classifier.html](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- **Gender-and-Age-Detection Github**：Sample Project  
  https://github.com/smahesh29/Gender-and-Age-Detection  
- **UTKFace 資料集 (原始版)**：20,000+ 張人臉影像，年齡範圍 0–116 歲，可作為年齡與性別預測訓練用。  
  https://www.kaggle.com/datasets/jangedoo/utkface-new 
