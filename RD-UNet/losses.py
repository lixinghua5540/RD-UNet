from __future__ import absolute_import
import tensorflow as tf
import tensorflow.keras.backend as K

smooth = 0.0000001

def jacc_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth))

def ms_ssim(y_true, y_pred):
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1)

    return 1 - tf_ms_ssim

def binary_focal_loss(y_true, y_pred,gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)

def mloss(y_true, y_pred):
    fl=binary_focal_loss(y_true, y_pred)
    ms=ms_ssim(y_true, y_pred)
    iou=jacc_coef(y_true, y_pred)
    cross = K.binary_crossentropy(y_true, y_pred)
    return fl+ms+cross+iou


