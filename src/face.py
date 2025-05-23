

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


# VGGFace 的預訓練權重檔（notop ResNet50）
BASE_WEIGHTS_PATH='./vggface_weights/rcmalli_vggface_tf_notop_resnet50.h5'

'''
create new networks based on vggface(resnet50 inside)
you can also use vgg16 or senet50 inside
'''
def Face(train=False):
    base_model = ResNet50(input_shape=(200, 200, 3), include_top=False, weights=None, pooling='avg')

    if train:
        # 載入 VGGFace 原始權重，再凍結大部分層
        base_model.load_weights(BASE_WEIGHTS_PATH, by_name=True, skip_mismatch=True)                      
        for layer in base_model.layers[:-50]:
            layer.trainable = False   
    
    
    base_output = base_model.output
    # age 1~93, treat age as classifications task
    output_a = Dense(93, activation='softmax', name='predications_age')(base_output)
    # gender 0 or 1
    output_g = Dense(2, activation='softmax', name='predications_gender')(base_output)
    # race 0~4
    output_r = Dense(5, activation='softmax', name='predications_race')(base_output)

    new_model = Model(inputs=base_model.input, outputs=[output_a, output_g, output_r], name='network_based_vggface')

    return new_model

if __name__ == '__main__':
    model = Face(train=True)
    for layer in model.layers:
        print('layer_name:{0}=====trainable:{1}'.format(layer.name, layer.trainable))
    model.summary()