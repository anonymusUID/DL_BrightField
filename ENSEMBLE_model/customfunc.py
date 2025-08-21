import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


# --- Custom Loss Functions and Metrics ---

def jaccard_score(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    y_pred = K.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred, bce_weight=0.5, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    bce = K.binary_crossentropy(y_true, y_pred)
    return bce_weight * bce + (1 - bce_weight) * dice

def weighted_binary_crossentropy(y_true, y_pred):
    weight = 10
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = weight * y_true * bce + (1 - y_true) * bce
    return K.mean(weighted_bce)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * K.pow(y_pred, gamma) * (1 - y_true)
        loss = weight * cross_entropy
        return K.mean(loss)
    return focal_loss_fixed

def boundary_loss(y_true, y_pred):
    def laplacian_kernel():
        return tf.constant([[0., 1., 0.],
                            [1., -4., 1.],
                            [0., 1., 0.]], shape=(3, 3, 1, 1), dtype=tf.float32)

    kernel = laplacian_kernel()
    y_true_edge = tf.nn.conv2d(y_true, kernel, strides=1, padding='SAME')
    y_pred_edge = tf.nn.conv2d(y_pred, kernel, strides=1, padding='SAME')
    diff = tf.abs(y_true_edge - y_pred_edge)
    return K.mean(diff)

def focal_dice_boundary_loss(y_true, y_pred, alpha=0.25, gamma=2.0, bce_weight=0.3, boundary_weight=0.1):
    fl = focal_loss(gamma=gamma, alpha=alpha)(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    bl = boundary_loss(y_true, y_pred)
    return bce_weight * fl + (1 - bce_weight - boundary_weight) * dl + boundary_weight * bl


# --- Custom Objects for Model Loading ---
custom_objects = {
    'jaccard_score': jaccard_score,
    'recall_m': recall_m,
    'precision_m': precision_m,
    'f1_m': f1_m,
    'dice_loss': dice_loss,
    'dice_bce_loss': dice_bce_loss,
    'weighted_binary_crossentropy': weighted_binary_crossentropy,
    'focal_loss': focal_loss(),
    'boundary_loss': boundary_loss,
    'focal_dice_boundary_loss': focal_dice_boundary_loss
}





# --- Model DenseNet ---
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

# --- Custom Loss Functions and Metrics ---