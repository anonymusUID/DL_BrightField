import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import DenseNet121

def densenet_unet(input_shape=(256,256,3), num_classes=1, l2_reg=1e-4, dropout_rate=0.5):
    # Input and regularization setup
    inputs = Input(shape=input_shape)
    weight_decay = l2(l2_reg)
    
    # Base DenseNet121 model with frozen encoder
    base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Freeze all encoder layers
    for layer in base_model.layers:
        layer.trainable = False

    # Get DenseNet skip connections (verified layer names)
    s1 = base_model.get_layer("conv1_relu").output       # 128x128
    s2 = base_model.get_layer("pool2_relu").output      # 64x64
    s3 = base_model.get_layer("pool3_relu").output      # 32x32
    s4 = base_model.get_layer("pool4_relu").output      # 16x16
    bridge = base_model.get_layer("relu").output        # 8x8

    # Decoder block with all features
    def upsample_block(x, skip, filters, target_size, dropout_rate):
        x = Conv2DTranspose(filters, (3,3), strides=2, padding='same',
                          kernel_regularizer=weight_decay)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Resize for alignment
        x = Lambda(lambda img: tf.image.resize(img, target_size))(x)
        skip = Lambda(lambda img: tf.image.resize(img, target_size))(skip)
        
        x = Concatenate()([x, skip])
        
        x = Conv2D(filters, (3,3), padding='same',
                 kernel_regularizer=weight_decay)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # Decoder path with progressive dropout
    x = upsample_block(bridge, s4, 512, (16,16), dropout_rate*0.5)  # 16x16
    x = upsample_block(x, s3, 256, (32,32), dropout_rate*0.6)       # 32x32
    x = upsample_block(x, s2, 128, (64,64), dropout_rate*0.7)       # 64x64
    
    # Final upsampling to 128x128
    x = Conv2DTranspose(64, (3,3), strides=2, padding='same',
                      kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Lambda(lambda img: tf.image.resize(img, (128,128)))(x)
    
    # Connect first skip
    skip1 = Lambda(lambda img: tf.image.resize(img, (128,128)))(s1)
    x = Concatenate()([x, skip1])
    
    # Final convolutions
    x = Conv2D(64, (3,3), padding='same',
             kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate*0.8)(x)
    
    # Upsample to final resolution
    x = Conv2DTranspose(32, (3,3), strides=2, padding='same',
                      kernel_regularizer=weight_decay)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda img: tf.image.resize(img, (256,256)))(x)
    
    # Output layer
    outputs = Conv2D(num_classes, (1,1), activation='sigmoid')(x)

    return Model(inputs=inputs, outputs=outputs)