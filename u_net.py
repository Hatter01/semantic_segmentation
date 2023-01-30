import tensorflow as tf
from tensorflow.keras.layers import *

def double_conv_block(conv, n_filters):
    # Conv2D then ReLU activation
    conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(conv)
    # Conv2D then ReLU activation
    conv = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(conv)
    return conv

def unet(img_size):
    # Encoder
    inputs = Input(shape=(img_size, img_size, 3))
    c1 = double_conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.3)(p1)
    
    c2 = double_conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.3)(p2)
    
    c3 = double_conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.3)(p3)
    
    c4 = double_conv_block(p3, 512)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.3)(p4)
    
    # Bottle neck
    c5 = double_conv_block(p4, 1024)
    
    # Decoder
    u6 = Conv2DTranspose(512, (3, 3), strides = (2, 2), padding = 'same', activation = "relu")(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(0.3)(u6)
    c6 = double_conv_block(u6, 512)
    
    u7 = Conv2DTranspose(256, (3, 3), strides = (2, 2), padding = 'same', activation = "relu")(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(0.3)(u7)
    c7 = double_conv_block(u7, 256)
    
    u8 = Conv2DTranspose(128, (3, 3), strides = (2, 2), padding = 'same', activation = "relu")(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(0.3)(u8)
    c8 = double_conv_block(u8, 128)
    
    u9 = Conv2DTranspose(64, (3, 3), strides = (2, 2), padding = 'same', activation = "relu")(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(0.3)(u9)
    c9 = double_conv_block(u9, 64)
    
    outputs = Conv2D(151, 1, activation='softmax')(c9)

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model