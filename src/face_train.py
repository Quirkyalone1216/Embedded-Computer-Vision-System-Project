
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
initial_lr=0.01
val_split=0.1
test_split=0.1
data_augmentation = True

train_data_path = '././././dataset/archive/UTKFace'
weights_output_path = './face_weights'

os.makedirs('results', exist_ok=True)

# learning rate schedule
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.2
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.04
        return self.initial_lr * 0.008

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
            image = cv2.imread(os.path.join(root_path, f))
            image = cv2.resize(image, (200, 200))
            x.append(image)
            y_a.append(int(f_items[0]) - 1)
            y_g.append(int(f_items[1]))
            y_r.append(int(f_items[2]))
    
    y_a = utils.to_categorical(y_a, 93)
    y_g = utils.to_categorical(y_g, 2)
    y_r = utils.to_categorical(y_r, 5)
    
    x = np.array(x)
    y_a = np.array(y_a)
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
    total_loss, age_loss, gender_loss, race_loss, age_acc, gender_acc, race_acc = (
        results[0], results[1], results[2], results[3], results[4], results[5], results[6]
    )

    # 取得預測結果
    preds = model.predict(x_test, batch_size=32, verbose=0)
    pred_age    = np.argmax(preds[0], axis=1)
    pred_gender = np.argmax(preds[1], axis=1)
    pred_race   = np.argmax(preds[2], axis=1)

    true_age    = np.argmax(y_test_a, axis=1)
    true_gender = np.argmax(y_test_g, axis=1)
    true_race   = np.argmax(y_test_r, axis=1)

    # ===== 新增：計算 MAE / MSE / RMSE =====
    true_age_year = true_age + 1
    pred_age_year = pred_age + 1
    mae_age = mean_absolute_error(true_age_year, pred_age_year)
    mse_age = mean_squared_error(true_age_year, pred_age_year)
    rmse_age = np.sqrt(mse_age)


    # 建立字典來儲存最終指標，並存成 JSON
    metrics = {
        'age': {
            'loss': age_loss,
            'accuracy': age_acc,
            'MAE': float(mae_age),
            'MSE': float(mse_age),
            'RMSE': float(rmse_age)
        },
        'gender': {'loss': gender_loss, 'accuracy': gender_acc},
        'race':   {'loss': race_loss,   'accuracy': race_acc}
    }
    with open('results/test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 性別 & 人種的混淆矩陣
    cm_gender = confusion_matrix(true_gender, pred_gender)
    cm_race   = confusion_matrix(true_race, pred_race)

    # 將 classification report 也輸出成 CSV
    cr_age    = classification_report(true_age, pred_age, output_dict=True)
    cr_gender = classification_report(true_gender, pred_gender, output_dict=True)
    cr_race   = classification_report(true_race, pred_race, output_dict=True)

    # 將 classification report 存成 JSON
    with open('results/cr_age.json', 'w', encoding='utf-8') as f:
        json.dump(cr_age, f, ensure_ascii=False, indent=2)
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


    opt = Adam(lr=initial_lr)
    #opt = SGD(lr=initial_lr, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss=['categorical_crossentropy','categorical_crossentropy', 'categorical_crossentropy'],
                  metrics=['accuracy'])    

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
            histogram_freq=1,
            batch_size=batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,      # 關閉影像輸出，避免無效 tensor shape 錯誤
            update_freq=500
        )
    ]
    
    if not data_augmentation:
        history = model.fit(
            x_train, [y_train_a, y_train_g, y_train_r],
            batch_size=batch_size,
            epochs=nb_epochs,
            callbacks=callbacks,
            validation_data=(x_test, [y_test_a, y_test_g, y_test_r])
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
            verbose=1,
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
