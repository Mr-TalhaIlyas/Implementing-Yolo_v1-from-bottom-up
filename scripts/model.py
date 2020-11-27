import tensorflow as tf
from losses import Loss
import numpy as np
if tf.__version__ == '2.0.0' or tf.__version__ == '2.2.0' or tf.__version__ == '2.2.0-rc2':
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2, l1
    from tensorflow.keras.applications import Xception, MobileNetV2
    from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D, Dense, LeakyReLU
    from tensorflow.keras.layers import MaxPooling2D, concatenate, Dropout, Flatten, Reshape, Concatenate
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
    from tensorflow.keras.backend import resize_images, int_shape
    import tensorflow.keras.backend as K
if tf.__version__ == '1.15.0' or tf.__version__ == '1.13.1':
    from keras.models import Model
    from keras.regularizers import l2, l1
    from keras.applications.mobilenetv2 import MobileNetV2
    from keras.applications.xception import Xception
    from keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D, PReLU, Lambda, add
    from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, Dropout, Flatten, Dense, Reshape
    from keras.layers import MaxPooling2D, concatenate, Concatenate, LeakyReLU, SeparableConv2D
    from keras.backend import resize_images, int_shape
    from keras.optimizers import Adam, Nadam, SGD
    import keras.backend as K
#%%

def yolo_output(FC2, grid_size, num_class, boxes_percell):
    
    S = grid_size
    C = num_class
    B = boxes_percell
    
    idx1 = S * S * C
    idx2 = idx1 + S * S * B
    
    #class_pr = Reshape((S, S, C))(FC2[:, :idx1])
    #obj_score = Reshape((S, S, B))(FC2[:, idx1:idx2])
    #b_box_c = Reshape((S, S, B*4))(FC2[:, idx2:])
    # class probabilities
    class_pr = K.reshape(FC2[:, :idx1], (K.shape(FC2)[0],) + tuple([S, S, C]))
    class_pr = K.softmax(class_pr)

    #confidence
    obj_score = K.reshape(FC2[:, idx1:idx2], (K.shape(FC2)[0],) + tuple([S, S, B]))
    obj_score = K.sigmoid(obj_score)

    # boxes
    b_box_c = K.reshape(FC2[:, idx2:], (K.shape(FC2)[0],) + tuple([S, S, B * 4]))
    b_box_c = K.sigmoid(b_box_c)

    # Get concatinated output for loss
    outputs = Concatenate(axis=-1)([class_pr, obj_score, b_box_c])
    
    return outputs
#%%
def yolo_output2(FC2, grid_size, num_class, boxes_percell):
    
    outputs = Reshape((grid_size, grid_size, ((boxes_percell * 5) + num_class)))(FC2)
    
    # Now reshaping each part of op_tensor accordingly
    class_pr = Lambda(lambda x: x[:,:,:, 0:num_class],
                      output_shape = (grid_size, grid_size, num_class), name = 'class_prob')(outputs)
    obj_score = Lambda(lambda x: x[:,:,:, num_class:num_class + boxes_percell],
                      output_shape = (grid_size, grid_size, boxes_percell), name = 'obj_score')(outputs)
    b_box_c = Lambda(lambda x: x[:,:,:, num_class + boxes_percell:],
                        output_shape = (grid_size, grid_size, boxes_percell*4), name = 'B_box_cord')(outputs)
    
    # Get concatinated output for loss
    outputs = Concatenate(axis=-1)([class_pr, obj_score, b_box_c])
    
    return outputs
#%%
def yolo(input_img, grid_size, boxes_percell, num_class, dropout, batchnorm = True, Alpha = 0.1):
    
    # Contracting Path
    c1 = Conv2D(filters = 64, kernel_size = (7, 7), strides=(2, 2), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = LeakyReLU(alpha=Alpha)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(filters = 192, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = LeakyReLU(alpha=Alpha)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(filters = 128, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=Alpha)(c3)
    c3 = Conv2D(filters = 256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=Alpha)(c3)
    c3 = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=Alpha)(c3)
    c3 = Conv2D(filters = 512, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = LeakyReLU(alpha=Alpha)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    #p3 = Dropout(dropout)(p3)
    
    for i in range(4):
        c4 = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = LeakyReLU(alpha=Alpha)(c4)
        c4 = Conv2D(filters = 512, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
        c4 = BatchNormalization()(c4)
        c4 = LeakyReLU(alpha=Alpha)(c4)
        
    c4 = Conv2D(filters = 512, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=Alpha)(c4)
    c4 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = LeakyReLU(alpha=Alpha)(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    #p4 = Dropout(dropout)(p4)

    for i in range(2):
        c5 = Conv2D(filters = 512, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = LeakyReLU(alpha=Alpha)(c5)
        c5 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
        c5 = BatchNormalization()(c5)
        c5 = LeakyReLU(alpha=Alpha)(c5)
        
    c5 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU(alpha=Alpha)(c5)
    c5 = Conv2D(filters = 1024, kernel_size = (3, 3), strides=(2, 2), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = LeakyReLU(alpha=Alpha)(c5)
    #c5 = Dropout(dropout)(c5)
    
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU(alpha=Alpha)(c6)
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = LeakyReLU(alpha=Alpha)(c6)
    #c6 = Dropout(dropout)(c6)

    
    flat = Flatten()(c6)
    FC1 = Dense(4096)(flat)
    FC1 = LeakyReLU(alpha=Alpha)(FC1)
    FC1 = Dropout(dropout)(FC1)
    fc2_nodes = grid_size * grid_size * ((boxes_percell * 5) + num_class)
    FC2 = Dense(fc2_nodes)(FC1) 
        
    outputs_a = yolo_output(FC2, grid_size, num_class, boxes_percell)
    
    model = Model(inputs=[input_img], outputs=[outputs_a])
    return model


#%%
def yolo_mine(input_img, grid_size, boxes_percell, num_class, dropout, batchnorm = True, Alpha = 0.1):
    # Entry FLow
    #input
    ip = Conv2D(filters = 32, kernel_size = (3,3), kernel_initializer = 'he_normal',  use_bias=False, strides = (2,2), padding = 'same')(input_img)
    ip = BatchNormalization()(ip)
    ip = LeakyReLU(alpha=Alpha)(ip)
    ip = Conv2D(filters = 64, kernel_size = (3,3), kernel_initializer = 'he_normal', use_bias=False, padding = 'same')(ip)
    ip = BatchNormalization()(ip)
    ip = LeakyReLU(alpha=Alpha)(ip)       # *******___1/2 times smaller than ip___********
    # 1st Residual connection
    res1 = Conv2D(filters = 128, kernel_size = (1,1), kernel_initializer = 'he_normal', use_bias=False, strides = (2,2), padding = 'same')(ip)
    res1 = BatchNormalization()(res1)
    # Block 1
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(ip)
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU(alpha=Alpha)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), dilation_rate = 1, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU(alpha=Alpha)(b1)
    b1 = SeparableConv2D(filters = 128, kernel_size = (3, 3), strides = (2,2), use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b1 = BatchNormalization()(b1)
    b1 = LeakyReLU(alpha=Alpha)(b1)       # *******___1/4 times smaller than ip___********
    b1 = add([b1, res1])
    # 2nd Residual connection
    res2 = Conv2D(filters = 256, kernel_size = (2,2), kernel_initializer = 'he_normal', use_bias=False, strides = (2,2))(b1)
    res2 = BatchNormalization()(res2)
    # Block 2
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b1)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU(alpha=Alpha)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), dilation_rate = 2, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU(alpha=Alpha)(b2)
    b2 = SeparableConv2D(filters = 256, kernel_size = (3, 3), strides = (2,2), use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b2 = BatchNormalization()(b2)
    b2 = LeakyReLU(alpha=Alpha)(b2)       # *******___1/8 times smaller than ip___********
    b2 = add([b2, res2])
    # 3rd Residual connection
    res3 = Conv2D(filters = 768, kernel_size = (2,2), kernel_initializer = 'he_normal', strides = (2,2), use_bias=False, padding = 'same')(b2)
    res3 = BatchNormalization()(res3)
    # Block 3
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b2)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU(alpha=Alpha)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 4, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU(alpha=Alpha)(b3)
    b3 = SeparableConv2D(filters = 768, kernel_size = (3, 3), strides = (2,2), use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3)
    b3 = BatchNormalization()(b3)
    b3 = LeakyReLU(alpha=Alpha)(b3)       # *******___1/16 times smaller than ip___********
    b3 = add([b3, res3])
    # Middle Flow
    # 4th residual connection  8
    res4 = b3
    b3_ = b3
    for i in range(8):
        
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b3_)
        b4 = BatchNormalization()(b4)
        b4 = LeakyReLU(alpha=Alpha)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = LeakyReLU(alpha=Alpha)(b4)
        b4 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 8, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
        b4 = BatchNormalization()(b4)
        b4 = LeakyReLU(alpha=Alpha)(b4)
        b4 = add([b4, res4])
        res4 = b4
        b3_ = b4 
    
    #Adding extra layer just to reduce parameters
    b4 = Conv2D(filters = 768, kernel_size = (3, 3), strides = (2,2), use_bias=False, kernel_initializer='he_normal', padding = 'same')(b4)
    b4 = BatchNormalization()(b4)
    b4 = LeakyReLU(alpha=Alpha)(b4)
    #b4 = Dropout(dropout)(b4)
    # Exit Flow
    
    # 5th residual connection
    res5 = Conv2D(filters = 1024, kernel_size = (1,1), strides = (2,2), use_bias=False, kernel_initializer = 'he_normal',  padding = 'same')(b4)
    res5 = BatchNormalization()(res5)
    # Block 5  2
    b5 = SeparableConv2D(filters = 768, kernel_size = (3, 3), dilation_rate = 1, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b4)
    b5 = BatchNormalization()(b5)
    b5 = LeakyReLU(alpha=Alpha)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), dilation_rate = 1, use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = LeakyReLU(alpha=Alpha)(b5)
    b5 = SeparableConv2D(filters = 1024, kernel_size = (3, 3), strides = (2,2), use_bias=False, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b5 = BatchNormalization()(b5)
    b5 = LeakyReLU(alpha=Alpha)(b5)      # *******___1/32 times smaller than ip___********
    b5 = Dropout(dropout)(b5)
    b5 = add([b5, res5])

    # Block 6
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), use_bias=False, dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b5)
    b6 = BatchNormalization()(b6)
    b6 = LeakyReLU(alpha=Alpha)(b6)
    b6 = SeparableConv2D(filters = 1536, kernel_size = (3, 3), use_bias=False, dilation_rate = 1, depthwise_initializer='he_normal', pointwise_initializer='he_normal',padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = LeakyReLU(alpha=Alpha)(b6)
    #b6 = Dropout(dropout)(b6)  
    b6 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='he_normal', pointwise_initializer='he_normal', padding = 'same')(b6)
    b6 = BatchNormalization()(b6)
    b6 = LeakyReLU(alpha=Alpha)(b6)

    # adding 2 FC layers
    flat = Flatten()(b6)
    FC1 = Dense(4096, use_bias=False, kernel_initializer='he_normal')(flat)
    FC1 = LeakyReLU(alpha=Alpha)(FC1)
    FC1 = Dropout(dropout)(FC1)
    fc2_nodes = grid_size * grid_size * ((boxes_percell * 5) + num_class)
    FC2 = Dense(fc2_nodes, use_bias=False, kernel_initializer='he_normal')(FC1) 
    
    
    outputs_a = yolo_output(FC2, grid_size, num_class, boxes_percell)
    
    model = Model(inputs=[input_img], outputs=[outputs_a])
    return model  
#%%

def yolo_exception(input_img, grid_size, boxes_percell, num_class, dropout, Alpha = 0.1, batchnorm = True):
    
    
    x_model = Xception(include_top = False, weights = 'imagenet', input_shape = input_img, pooling = None, classes = num_class)#"imagenet"
    for layer in x_model.layers:
        if isinstance(layer, Activation):# putting leaky relu activation in model
            layer.activation = tf.keras.activations.relu(layer.input, alpha=Alpha)
    
    res5 = x_model.output
    # adding 4 extra conv layers
    #res5 = Dropout(dropout)(res5)  
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), strides = (2,2), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = Dropout(dropout)(res5)
    # adding 2 FC layers
    flat = Flatten()(res5)
    FC1 = Dense(4096, use_bias=False, kernel_initializer='glorot_uniform')(flat)
    FC1 = LeakyReLU(alpha=Alpha)(FC1)
    FC1 = Dropout(dropout)(FC1)
    fc2_nodes = grid_size * grid_size * ((boxes_percell * 5) + num_class)
    FC2 = Dense(fc2_nodes, use_bias=False, kernel_initializer='glorot_uniform')(FC1) 
    
    outputs_a = Reshape((grid_size, grid_size, ((boxes_percell * 5) + num_class)))(FC2)
    #outputs_a = yolo_output(FC2, grid_size, num_class, boxes_percell)
    model = Model(inputs=x_model.input, outputs=[outputs_a])    
    return model
#%%
def yolo_exception_mod(input_img, grid_size, boxes_percell, num_class, dropout, Alpha = 0.1, batchnorm = True):
    
    
    x_model = Xception(include_top = False, weights = 'imagenet', input_shape = input_img, pooling = None, classes = num_class)#"imagenet"
    for layer in x_model.layers:
        if isinstance(layer, Activation):# putting leaky relu activation in model
            layer.activation = tf.keras.activations.relu(layer.input, alpha=Alpha)
    
    res5 = x_model.output
    # adding 4 extra conv layers
    res5 = Dropout(dropout)(res5)  
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), strides = (2,2), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = Dropout(dropout)(res5)
    # adding 2 FC layers
    flat = Flatten()(res5)
    FC1 = Dense(4096, use_bias=False, kernel_initializer='glorot_uniform')(flat)
    FC1 = LeakyReLU(alpha=Alpha)(FC1)
    FC1 = Dropout(dropout)(FC1)
    
    FC_pr = Dense((grid_size * grid_size * num_class))(FC1)
    C_pr = Reshape((grid_size, grid_size, num_class), name = 'class_prob')(FC_pr)
    
    FC_coord = Dense((grid_size * grid_size * boxes_percell * 4))(FC1)
    C_coord = Reshape((grid_size, grid_size, boxes_percell * 4), name = 'class_coord')(FC_coord)
    
    FC_obj = Dense((grid_size * grid_size * boxes_percell))(FC1)
    C_obj = Reshape((grid_size, grid_size, boxes_percell), name = 'class_obj')(FC_obj)
    
    outputs_a = Concatenate(axis=-1)([C_pr, C_obj, C_coord])
    #outputs_a = yolo_output(FC2, grid_size, num_class, boxes_percell)
    model = Model(inputs=x_model.input, outputs=[outputs_a])    
    return model

#%%
def yolo_mobilenet_v2(input_img, grid_size, boxes_percell, num_class, dropout, Alpha = 0.1, batchnorm = True):
    
    
    x_model = MobileNetV2(alpha=1.4, include_top = False, weights = 'imagenet', input_shape = input_img, pooling = None, classes = num_class)#"imagenet"
    for layer in x_model.layers:
        if isinstance(layer, Activation):# putting leaky relu activation in model
            layer.activation = tf.keras.activations.relu(layer.input, alpha=Alpha)
    
    res5 = x_model.output
    # adding 4 extra conv layers
    res5 = Dropout(dropout)(res5)  
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 2048, kernel_size = (3,3), strides = (2,2), use_bias=False, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = SeparableConv2D(filters = 512, kernel_size = (3,3), use_bias=False,  depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', padding = 'same')(res5)
    res5 = BatchNormalization()(res5)
    res5 = LeakyReLU(alpha=Alpha)(res5)
    res5 = Dropout(dropout)(res5)
    # adding 2 FC layers
    flat = Flatten()(res5)
    FC1 = Dense(4096, use_bias=False, kernel_initializer='glorot_uniform')(flat)
    FC1 = LeakyReLU(alpha=Alpha)(FC1)
    FC1 = Dropout(dropout)(FC1)
    # fc2_nodes = grid_size * grid_size * ((boxes_percell * 5) + num_class)
    # FC2 = Dense(fc2_nodes, use_bias=False, kernel_initializer='glorot_uniform')(FC1) 
    
    ########################
    FC_pr = Dense((grid_size * grid_size * num_class))(FC1)
    C_pr = Reshape((grid_size, grid_size, num_class), name = 'class_prob')(FC_pr)
    
    FC_coord = Dense((grid_size * grid_size * boxes_percell * 4))(FC1)
    C_coord = Reshape((grid_size, grid_size, boxes_percell * 4), name = 'class_coord')(FC_coord)
    
    FC_obj = Dense((grid_size * grid_size * boxes_percell))(FC1)
    C_obj = Reshape((grid_size, grid_size, boxes_percell), name = 'class_obj')(FC_obj)
    
    outputs_a = Concatenate(axis=-1)([C_pr, C_obj, C_coord])
    ###########################
    #outputs_a = Reshape((grid_size, grid_size, ((boxes_percell * 5) + num_class)))(FC2)
    #outputs_a = yolo_output(FC2, grid_size, num_class, boxes_percell)
    model = Model(inputs=x_model.input, outputs=[outputs_a])   
    return model

'''
    # Now activating each part of op_tensor accordingly
    class_pr = Lambda(lambda x: x[:,:,:, 0:num_class],
                      output_shape = (grid_size, grid_size, num_class), name = 'class_prob')(outputs)
    # class_pr = Lambda(lambda x: tf.keras.activations.softmax(x, axis=-1),
    #                   output_shape = (grid_size, grid_size, num_class), name = 'class_prob_a')(class_pr)
    
    # if op_activation:
    #     class_pr = Lambda(lambda x: tf.keras.activations.softmax(x, axis=-1),
    #                       output_shape = (grid_size, grid_size, num_class), name = 'class_prob_a')(class_pr)
    #     obj_score = Lambda(lambda x: tf.keras.activations.sigmoid(x),
    #                       output_shape = (grid_size, grid_size, boxes_percell), name = 'obj_score_a')(obj_score)
    #     b_box_c = Lambda(lambda x: tf.keras.activations.sigmoid(x),
    #                         output_shape = (grid_size, grid_size, boxes_percell*4), name = 'B_box_cord_a')(b_box_c)
    
'''




















#%%
'''
yolo_op_tenosr = Input((grid_size, grid_size, num_class+5), name='op')
input_tensor = [input_img, yolo_op_tenosr]

model = yolo_loss(input_tensor, grid_size, boxes_percell, num_class, dropout = 0.3, activation = 'relu', batchnorm = True)
    
'''
def yolo_loss(input_tensor, grid_size, boxes_percell, num_class, dropout, batchnorm = True, activation = 'relu', training=True):
    
    input_img = input_tensor[0]
    label_tensor = input_tensor[1]
    label_tensor = reverse_order(label_tensor, grid_size, num_class)
    input_shape = tf.convert_to_tensor(np.array(int_shape(input_img)[1:]))
    # Contracting Path
    c1 = Conv2D(filters = 64, kernel_size = (7, 7), kernel_initializer = 'he_normal', padding = 'same')(input_img)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation)(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(filters = 192, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation(activation)(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(filters = 128, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(filters = 256, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    c3 = Conv2D(filters = 512, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation(activation)(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    for i in range(4):
        c4 = Conv2D(filters = 256, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Activation(activation)(c4)
        c4 = Conv2D(filters = 512, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
        c4 = BatchNormalization()(c4)
        c4 = Activation(activation)(c4)
        
    c4 = Conv2D(filters = 512, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation(activation)(c4)
    c4 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation(activation)(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    for i in range(2):
        c5 = Conv2D(filters = 512, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Activation(activation)(c5)
        c5 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
        c5 = BatchNormalization()(c5)
        c5 = Activation(activation)(c5)
        
    c5 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation(activation)(c5)
    c5 = Conv2D(filters = 1024, kernel_size = (3, 3), strides=(2, 2), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation(activation)(c5)
    c5 = Dropout(dropout)(c5)
    
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c5)
    c6 = BatchNormalization()(c6)
    c6 = Activation(activation)(c6)
    c6 = Conv2D(filters = 1024, kernel_size = (3, 3), kernel_initializer = 'he_normal', padding = 'same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation(activation)(c6)
    c6 = Dropout(dropout)(c6)

    
    flat = Flatten()(c6)
    FC1 = Dense(4096)(flat)
    FC1 = Dropout(dropout)(FC1)
    fc2_nodes = grid_size * grid_size * ((boxes_percell * 5) + num_class)
    FC2 = Dense(fc2_nodes, activation='sigmoid')(FC1) 
    
    coord_loss, object_loss, noobject_loss, cls_loss = Loss()([FC2, label_tensor, input_shape])
        
    # Now activating each part of op_tensor accordingly
    index_classification = tf.multiply(tf.pow(grid_size, 2), num_class)
    index_confidence = tf.multiply(tf.pow(grid_size, 2), num_class + boxes_percell)
    
    class_pr = Reshape((grid_size, grid_size, num_class))(FC2[:, :index_classification])
    #class_pr = tf.reshape(FC2[:, :index_classification], [-1, grid_size, grid_size, num_class])
    class_pr = Lambda(lambda x: tf.keras.activations.softmax(x, axis=-1),
                      output_shape = (grid_size, grid_size, num_class), name = 'class_prob_a')(class_pr)
    
    obj_score = Reshape((grid_size, grid_size, boxes_percell))(FC2[:, index_classification:index_confidence])
    #obj_score = tf.reshape(FC2[:, index_classification:index_confidence], [-1, grid_size, grid_size, boxes_percell])
    obj_score = Lambda(lambda x: tf.keras.activations.sigmoid(x),
                      output_shape = (grid_size, grid_size, boxes_percell), name = 'obj_score_a')(obj_score)
    
    b_box_c = Reshape((grid_size, grid_size, boxes_percell * 4))(FC2[:, index_confidence:])
    #b_box_c = tf.reshape(FC2[:, index_confidence:], [-1, grid_size, grid_size, boxes_percell * 4])
    b_box_c = Lambda(lambda x: tf.keras.activations.sigmoid(x),
                        output_shape = (grid_size, grid_size, boxes_percell*4), name = 'B_box_cord_a')(b_box_c)
    
    outputs = Concatenate(axis=-1)([class_pr, b_box_c, obj_score])
    # if training == True:
    #     outputs = coord_loss, object_loss, noobject_loss, cls_loss, outputs
    # else:
    
    
    
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def reverse_order(label_tensor, grid_size, num_class):
    
    class_pr = label_tensor[:,:,:, 0:num_class]
    b_box    = label_tensor[:,:,:, num_class:num_class+4] 
    obj_score= label_tensor[:,:,:, num_class+4:] 
    label_tensor = Concatenate(axis=-1)([obj_score, b_box, class_pr])
    
    return label_tensor