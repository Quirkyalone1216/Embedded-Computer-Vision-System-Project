from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation

# VGGFace 的預訓練權重檔（notop ResNet50）
BASE_WEIGHTS_PATH = './vggface_weights/rcmalli_vggface_tf_notop_resnet50.h5'

def Face(train=False):
    """
    建立以 ResNet50 (notop) 為 backbone，三個任務 (age/regression, gender, race) 的多工網路。
    關鍵修改：
      1. 使用 pooling='avg' 取出 feature vector 後，插入一層 FC (Dense→BN→ReLU→Dropout) 作為 shared embedding，
         再從這個 shared embedding 分岔到三個 head。
      2. 將年齡分支改成迴歸 (output 1 個神經元)，loss 改用 MAE/MSE 會更合理；若要維持分類，也可將 Dense(93) 保留，但以下範例示範迴歸做法。
      3. 頭部都加上 Dropout 以減少過擬合，並加 BatchNormalization 幫助穩定訓練。
    """
    # 1. 載入 ResNet50 backbone，不包含最上層 FC，並取 avg pooling
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,          # 先不載入 weights，再用 load_weights by_name 來讀入 VGGFace 的權重
        pooling='avg'
    )

    if train:
        # 2. 載入 VGGFace (ResNet50-notop) 的原始權重
        base_model.load_weights(BASE_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        # 3. 凍結 backbone 前面大部分層，只微調最後 30 層
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers[-30:]:
            layer.trainable = True

    # 4. 從 backbone 拿到 pooling 後的向量
    x = base_model.output  # shape: (None, 2048)

    # 5. 在 shared embedding 上加一個 FC block：Dense→BN→ReLU→Dropout
    x = Dense(1024, name='shared_fc1')(x)
    x = BatchNormalization(name='shared_bn1')(x)
    x = Activation('relu', name='shared_relu1')(x)
    x = Dropout(0.5, name='shared_dropout1')(x)
    # 現在 x 的 shape = (None, 512)

    # --- 6. 年齡分支 (保持迴歸，只輸出一個連續值) ---
    # 直接由 shared embedding 分出一個 Dense→BN→ReLU→Dropout，再輸出迴歸預測
    a = Dense(256, name='age_fc1')(x)
    a = BatchNormalization(name='age_bn1')(a)
    a = Activation('relu', name='age_relu1')(a)
    a = Dropout(0.5, name='age_dropout1')(a)
    # 改成 94 類別分類 (0~93 歲)，再在訓練時用 one‐hot 並計算預測年齡
    output_a = Dense(94, activation='softmax', name='predictions_age')(a)

    # --- 7. 性別分支 (二分類) ---
    g = Dense(256, name='gender_fc1')(x)
    g = BatchNormalization(name='gender_bn1')(g)
    g = Activation('relu', name='gender_relu1')(g)
    g = Dropout(0.5, name='gender_dropout1')(g)
    g = Dense(128, name='gender_fc2')(g)
    g = BatchNormalization(name='gender_bn2')(g)
    g = Activation('relu', name='gender_relu2')(g)
    g = Dropout(0.5, name='gender_dropout2')(g)
    output_g = Dense(2, activation='softmax', name='predictions_gender')(g)

    # --- 8. 種族分支 (5 分類) ---
    r = Dense(256, name='race_fc1')(x)
    r = BatchNormalization(name='race_bn1')(r)
    r = Activation('relu', name='race_relu1')(r)
    r = Dropout(0.5, name='race_dropout1')(r)
    r = Dense(128, name='race_fc2')(r)
    r = BatchNormalization(name='race_bn2')(r)
    r = Activation('relu', name='race_relu2')(r)
    r = Dropout(0.5, name='race_dropout2')(r)
    output_r = Dense(5, activation='softmax', name='predictions_race')(r)

    # 9. 整合三個輸出
    new_model = Model(
        inputs=base_model.input,
        outputs=[output_a, output_g, output_r],
        name='network_based_vggface'
    )

    return new_model


if __name__ == '__main__':
    model = Face(train=True)
    for layer in model.layers:
        print(f'layer_name: {layer.name} ===== trainable: {layer.trainable}')
    model.summary()
