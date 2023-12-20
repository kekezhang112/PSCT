from __future__ import division
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda,GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dropout, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def M_VGG16SR(input_tensor=None):
    input_shape = (None, None,3)

    if input_tensor is None:
        img_inputSR = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_inputSR = Input(tensor=input_tensor)
        else:
            img_inputSR = input_tensor

    # stage1
    conv1_1SR = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(img_inputSR)
    conv1_2SR = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1SR)
    conv1_poolSR = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2SR)

    # stage2
    conv2_1SR = Convolution2D(128,kernel_size=(3, 3), activation='relu', padding='same')(conv1_poolSR)
    conv2_2SR = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1SR)
    conv2_poolSR = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2SR)

    # stage3
    conv3_1SR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv2_poolSR)
    conv3_2SR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_1SR)
    conv3_3SR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_2SR)
    conv3_poolSR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv3_3SR)

    # stage4
    conv4_1SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv3_poolSR)
    conv4_2SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_1SR)
    conv4_3SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_2SR)
    conv4_poolSR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv4_3SR)

    # stage5
    conv5_1SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_poolSR)
    conv5_2SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv5_1SR)
    conv5_3SR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv5_2SR)
    conv5_poolSR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv5_3SR)

    model = Model(inputs=[img_inputSR], outputs=[conv1_poolSR,conv2_poolSR,conv3_poolSR,conv4_poolSR,conv5_poolSR], name='M_VGG16SR')

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

    model.load_weights(weights_path)

    return model


def M_VGG16HR(input_tensor=None):
    input_shape = (None, None, 3)

    if input_tensor is None:
        img_inputHR = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_inputHR = Input(tensor=input_tensor)
        else:
            img_inputHR = input_tensor

    # stage1
    conv1_1HR = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(img_inputHR)
    conv1_2HR = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding='same')(conv1_1HR)
    conv1_poolHR = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2HR)

    # stage2
    conv2_1HR = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv1_poolHR)
    conv2_2HR = Convolution2D(128, kernel_size=(3, 3), activation='relu', padding='same')(conv2_1HR)
    conv2_poolHR = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2HR)

    # stage3
    conv3_1HR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv2_poolHR)
    conv3_2HR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_1HR)
    conv3_3HR = Convolution2D(256, kernel_size=(3, 3), activation='relu', padding='same')(conv3_2HR)
    conv3_poolHR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv3_3HR)

    # stage4
    conv4_1HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv3_poolHR)
    conv4_2HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_1HR)
    conv4_3HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(conv4_2HR)
    conv4_poolHR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv4_3HR)

    # stage5
    conv5_1HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(
        conv4_poolHR)
    conv5_2HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(
        conv5_1HR)
    conv5_3HR = Convolution2D(512, kernel_size=(3, 3), activation='relu', padding='same')(
        conv5_2HR)
    conv5_poolHR = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(conv5_3HR)

    model = Model(inputs=[img_inputHR], outputs=[conv1_poolHR,conv2_poolHR,conv3_poolHR,conv4_poolHR,conv5_poolHR], name='M_VGG16HR')

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')

    model.load_weights(weights_path)

    return model


def PSCTmodel(img_rows,img_cols):

    inputimageSR = Input(shape=(img_rows, img_cols,3))
    inputimageHR = Input(shape=(img_rows, img_cols,3))
    

    base_modelSR = M_VGG16SR(input_tensor=inputimageSR)
    base_modelHR = M_VGG16HR(input_tensor=inputimageHR)

    FHR_1, FHR_2, FHR_3, FHR_4, FHR_5 = base_modelHR.output
    FSR_1, FSR_2, FSR_3, FSR_4, FSR_5 = base_modelSR.output

    f1 = layers.subtract([FHR_1, FSR_1])
    f2 = layers.subtract([FHR_2, FSR_2])
    f3 = layers.subtract([FHR_3, FSR_3])
    f4 = layers.subtract([FHR_4, FSR_4])
    f5 = layers.subtract([FHR_5, FSR_5])

    g1 = GlobalAveragePooling2D()(f1)
    g2 = GlobalAveragePooling2D()(f2)
    g3 = GlobalAveragePooling2D()(f3)
    g4 = GlobalAveragePooling2D()(f4)
    g5 = GlobalAveragePooling2D()(f5)

    fc11 = Dense(64, activation='relu', name='fc11')(g1)
    fc11 = Dropout(0.5)(fc11)
    fc12 = Dense(64, activation='relu', name='fc12')(fc11)

    fc21 = Dense(128, activation='relu', name='fc21')(g2)
    fc21 = Dropout(0.5)(fc21)
    fc22 = Dense(128, activation='relu', name='fc22')(fc21)

    fc31 = Dense(256, activation='relu', name='fc31')(g3)
    fc31 = Dropout(0.5)(fc31)
    fc32 = Dense(256, activation='relu', name='fc32')(fc31)

    fc41 = Dense(512, activation='relu', name='fc41')(g4)
    fc41 = Dropout(0.5)(fc41)
    fc42 = Dense(512, activation='relu', name='fc42')(fc41)

    fc51 = Dense(512, activation='relu', name='fc51')(g5)
    fc51 = Dropout(0.5)(fc51)
    fc52 = Dense(512, activation='relu', name='fc52')(fc51)

    s1 = Dense(1, name='ss1')(fc12)
    s2 = Dense(1, name='ss2')(fc22)
    s3 = Dense(1, name='ss3')(fc32)
    s4 = Dense(1, name='ss4')(fc42)
    s5 = Dense(1, name='ss5')(fc52)

    s_s = layers.average([s1,s2,s3,s4,s5],name='s_s')

    fraw = FSR_5
    gwn = GlobalAveragePooling2D()(fraw)
    wx = Dense(512, activation='relu', name='fcw1')(gwn)
    wx = Dropout(0.5)(wx)
    wx = Dense(256, activation='relu', name='fcw2')(wx)
    w_c = Dense(1, name='w_c')(wx)

    inputSaliency = Input(shape=(None, None, 1))
    fsm = layers.multiply([fraw, inputSaliency])
    gn = GlobalAveragePooling2D()(fsm)

    x = Dense(512, activation='relu', name='fc1')(gn)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    s_c = Dense(1, name='s_c')(x)

    finalscore = Lambda(lambda x: x[0]*(1-x[2])+x[1]*x[2])([s_s,s_c,w_c])
    model = Model(inputs=[inputimageSR,inputimageHR,inputSaliency], outputs=[finalscore])

    return model

