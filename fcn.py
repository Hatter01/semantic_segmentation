import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications.vgg16 import *

def fcn(img_size, ch_out=151):
    
    inputs = Input(shape=(img_size, img_size, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    f3 = vgg16.get_layer('block3_pool').output  
    f4 = vgg16.get_layer('block4_pool').output  
    f5 = vgg16.get_layer('block5_pool').output  

    f5_c1 = Conv2D(filters=4086, kernel_size=7, padding='same', activation='relu')(f5)
    f5_d1 = Dropout(0.3)(f5_c1)
    f5_c2 = Conv2D(filters=4086, kernel_size=1, padding='same', activation='relu')(f5_d1)
    f5_d2 = Dropout(0.3)(f5_c2)
    f5_c3 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f5_d2)

    f5_c3_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2, use_bias=False, padding='same', activation='relu')(f5)
    f4_c1 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f4)

    m1 = add([f4_c1, f5_c3_x2])

    m1_x2 = Conv2DTranspose(filters=ch_out, kernel_size=4, strides=2, use_bias=False, padding='same', activation='relu')(m1)
    f3_c1 = Conv2D(filters=ch_out, kernel_size=1, padding='same', activation=None)(f3)
    m2 = add([f3_c1, m1_x2])

    outputs = Conv2DTranspose(filters=ch_out, kernel_size=16, strides=8, padding='same', activation=None)(m2)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model
