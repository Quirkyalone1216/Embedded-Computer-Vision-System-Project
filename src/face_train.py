
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


if __name__ == '__main__':
    train()
