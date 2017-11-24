#!/usr/bin/env python3
# encoding: utf-8

#from keras.applications import vgg16
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

def VGG16(num_classes, activation="relu"):
#    base_model = vgg16.VGG16(weights='imagenet', include_top=False)
#    x = base_model.output

    img_input = Input(shape=(None, None, 1))

    # Block 1
    x = Conv2D(64, (3, 3), activation=activation, padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation=activation, padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation=activation, padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation=activation, padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation=activation, padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation=activation, padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation=activation, padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation=activation, padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    x = Conv2D(1024, (1, 1), activation=activation, padding='same', name='block6_conv1')(x)
    
    # Global Pooling
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(1024, activation=activation)(x)
    x = Dense(1024, activation=activation)(x)
    
    x = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=img_input, outputs=x)