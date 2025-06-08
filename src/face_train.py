
'''
train on my own dataset
'''
from face import Face

import numpy as np
np.object = object
np.bool = bool
np.int = int

from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2


import time
import utils as my_utils
import urllib.request

from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error
import json

from random_eraser import get_random_eraser
from mixup_generator import MixupGenerator

# config for training
batch_size=32
nb_epochs=30
initial_lr=0.001
val_split=0.1
test_split=0.1
data_augmentation = True

train_data_path = '././././dataset/archive/UTKFace'
weights_output_path = './face_weights'

os.makedirs('results', exist_ok=True)

# 年齡總共 0~93 => 94 個類別
NUM_AGE_CLASSES = 94

# learning rate schedule
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.2
        return self.initial_lr * 0.05

# load train data from disk
# or you can use fit_generator(...) instead of fit(...)
def load_data():
    x = []
    y_a = []
    y_g = []
    y_r = []

    # loop the images
    root_path, dirs, files = next(os.walk(train_data_path))

    for f in files:
        f_items = str(f).split('_')
        # age between 1 and 93
        if len(f_items) == 4 and int(f_items[0]) <= 93:
            # 讀取後先將 BGR → RGB，再調整尺寸
            image = cv2.imread(os.path.join(root_path, f))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            x.append(image)
            y_a.append(int(f_items[0]))
            y_g.append(int(f_items[1]))
            y_r.append(int(f_items[2]))
    
    y_g = utils.to_categorical(y_g, 2)
    y_r = utils.to_categorical(y_r, 5)
    
    x = np.array(x)
    # 把 age 轉成 one‐hot (0~93 共 94 類)
    y_a_int = np.array(y_a, dtype=np.int32)
    y_a = to_categorical(y_a_int, NUM_AGE_CLASSES)
    y_g = np.array(y_g)
    y_r = np.array(y_r)

    # shuffle the indexs
    indexs = np.arange(len(x))
    np.random.shuffle(indexs)
    
    x = x[indexs]
    y_a = y_a[indexs]
    y_g = y_g[indexs]
    y_r = y_r[indexs]

    # preprocess
    x = my_utils.preprocess_input(x, data_format='channels_last', version=2)
    return x, y_a, y_g, y_r


# 新增：在測試集上計算並呈現各任務的最終指標，並輸出 CSV 及混淆矩陣圖片到 results
def evaluate_on_testset(model, x_test, y_test_a, y_test_g, y_test_r):
    # 評估 Loss 與 Accuracy
    results = model.evaluate(
        x_test, [y_test_a, y_test_g, y_test_r],
        batch_size=32, verbose=0
    )
    total_loss  = results[0]
    age_loss    = results[1]
    gender_loss = results[2]
    race_loss   = results[3]
    # evaluate() 回傳值為 [loss, loss1, loss2, loss3, acc1, acc2, acc3]
    age_acc     = results[4]
    gender_acc  = results[5]
    race_acc    = results[6]

    # 取得預測結果
    preds = model.predict(x_test, batch_size=32, verbose=0)
    # 年齡分類輸出 (shape=(N,94))，取「加權平均」或 argmax 還原成實際年齡
    age_probs   = preds[0]                # shape=(N,94)
    # 使用加權平均還原預測年齡
    pred_age    = np.dot(age_probs, np.arange(NUM_AGE_CLASSES))  # shape=(N,)
    pred_gender = np.argmax(preds[1], axis=1)
    pred_race   = np.argmax(preds[2], axis=1)

    # 真實年齡已為整數
    true_age    = np.argmax(y_test_a, axis=1)  # 從 one‐hot 還原成整數
    true_gender = np.argmax(y_test_g, axis=1)
    true_race   = np.argmax(y_test_r, axis=1)

    # ===== 新增：計算 MAE / MSE / RMSE =====
    # 迴歸直接比較「年齡值」，不再做 +1 轉換
    true_age_year = true_age
    pred_age_year = pred_age
    mae_age = mean_absolute_error(true_age_year, pred_age_year)
    mse_age = mean_squared_error(true_age_year, pred_age_year)
    rmse_age = np.sqrt(mse_age)


    # 建立字典來儲存最終指標，並存成 JSON
    metrics = {
        'age': {
            'loss': float(age_loss),
            # 使用自行計算的 mae_age / mse_age / rmse_age
            'MAE': float(mae_age),
            'MSE': float(mse_age),
            'RMSE': float(rmse_age)
        },
        'gender': {
            'loss': float(gender_loss),
            'accuracy': float(gender_acc)
        },
        'race': {
            'loss': float(race_loss),
            'accuracy': float(race_acc)
        }
    }
    with open('results/test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 年齡不做 classification report，只列 gender & race
    cr_gender = classification_report(true_gender, pred_gender, output_dict=True)
    cr_race   = classification_report(true_race,   pred_race,   output_dict=True)
    with open('results/cr_gender.json', 'w', encoding='utf-8') as f:
        json.dump(cr_gender, f, ensure_ascii=False, indent=2)
    with open('results/cr_race.json', 'w', encoding='utf-8') as f:
        json.dump(cr_race, f, ensure_ascii=False, indent=2)

    print("已將測試結果儲存至 results 資料夾")

'''
fine-tuning with UTKFace dataset
'''
def train():
    x, y_a, y_g, y_r = load_data()
    print(x.shape)
    print(y_a.shape)
    print(y_g.shape)
    print(y_r.shape)
    
    train_index = int(len(x)*(1-test_split))

    x_train = x[:train_index]
    y_train_a = y_a[:train_index]
    y_train_g = y_g[:train_index]
    y_train_r = y_r[:train_index]

    x_test = x[train_index:]
    y_test_a = y_a[train_index:]
    y_test_g = y_g[train_index:]
    y_test_r = y_r[train_index:]

    # --- BEGIN: 自動下載 VGGFace 權重 ---
    weights_dir  = './vggface_weights'
    weights_file = os.path.join(weights_dir,
                      'rcmalli_vggface_tf_notop_resnet50.h5')
    if not os.path.exists(weights_file):
        os.makedirs(weights_dir, exist_ok=True)
        url = ('https://github.com/rcmalli/keras-vggface/'
               'releases/download/v2.0/'
               'rcmalli_vggface_tf_notop_resnet50.h5')
        print(f'權重檔不存在，開始下載：{url}')
        urllib.request.urlretrieve(url, weights_file)
        print('下載完成，儲存至', weights_file)
    # --- END: 自動下載 VGGFace 權重 ---

    model = Face(train=True)

    opt = Adam(learning_rate=initial_lr)
    # age: 分類；gender & race: 分類
    model.compile(
        optimizer=opt,
        loss={
            'predictions_age': 'categorical_crossentropy',
            'predictions_gender': 'categorical_crossentropy',
            'predictions_race': 'categorical_crossentropy'
        },
        loss_weights={
            'predictions_age': 1.0,
            'predictions_gender': 0.5,
            'predictions_race': 2.0
        },
        metrics={
            'predictions_age': 'accuracy',
            'predictions_gender': 'accuracy',
            'predictions_race': 'accuracy'
        }
    )

    callbacks = [
        LearningRateScheduler(schedule=Schedule(nb_epochs, initial_lr)),
        ModelCheckpoint(
            filepath=os.path.join(weights_output_path, 
                                  "face_weights.{epoch:02d}-val_loss-{val_loss:.2f}.utk.h5"),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="auto"
        ),
        TensorBoard(
            log_dir=os.path.join('logs', f"{model.name}-{time.time()}"),
            histogram_freq=0,        # 關閉 histogram 以避免頻繁 profiler 輸出
            write_graph=True,
            write_grads=False,
            write_images=False,      # 關閉影像輸出，避免無效 tensor shape 錯誤
            update_freq='epoch',     # 每個 epoch 更新一次
            profile_batch=0          # 完全關閉 profiler
        )
    ]
    
    if not data_augmentation:
        history = model.fit(
            x_train, [y_train_a, y_train_g, y_train_r],
            batch_size=batch_size,
            epochs=nb_epochs,
            callbacks=callbacks,
            validation_data=(x_test, [y_test_a, y_test_g, y_test_r]),
            verbose=1  # 只顯示單一進度條
        )
    else:
        datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            preprocessing_function=get_random_eraser(v_l=0, v_h=255)
        )
        training_generator = MixupGenerator(
            x_train, [y_train_a, y_train_g, y_train_r],
            batch_size=batch_size, alpha=0.2,
            datagen=datagen
        )()
        history = model.fit(
            training_generator,
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_test, [y_test_a, y_test_g, y_test_r]),
            epochs=nb_epochs,
            verbose=1,  # 只顯示單一進度條
            callbacks=callbacks
        )

    # 把 history.history 內的 NumPy 型別都轉成純 Python 型別
    history_py = {}
    for key, values in history.history.items():
        # values 通常是 list of numpy.float32／numpy.int32…，把它們轉成 Python float/int
        history_py[key] = [float(v) for v in values]
    with open('results/history.json', 'w', encoding='utf-8') as f:
        json.dump(history_py, f, ensure_ascii=False, indent=2)

    # 在測試集上做最終評估並將結果輸出到 results
    evaluate_on_testset(model, x_test, y_test_a, y_test_g, y_test_r)


if __name__ == '__main__':
    train()
