import numpy as np

import keras
import keras.backend as K
import tensorflow as tf

# ---------------------------------------------------------------------------------------------------------------------
def dice_coef(y_true, y_pred, smooth=0.001):
    ''' Dice Coefficient

    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    '''

    class_num = 2
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

# ---------------------------------------------------------------------------------------------------------------------
def dice_coef_loss(y_true, y_pred):
    ''' Dice Coefficient Loss

    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    '''
    return 1-dice_coef(y_true, y_pred)

# ---------------------------------------------------------------------------------------------------------------------
def weighted_cross_entropy(y_true, y_pred, beta=0.9):
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = 1 - tf.cast(y_true, tf.float32)

    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)

# ---------------------------------------------------------------------------------------------------------------------
def weighted_bincrossentropy(true, pred, weight_zero=0.1, weight_one=1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.

    This can be useful for unbalanced catagories.

    Adjust the weights here depending on what is required.

    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
        will be penalize 10 times as much as false negatives.
    """
    true = float(true)
    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)

# ---------------------------------------------------------------------------------------------------------------------
def dyn_weighted_bincrossentropy(true, pred):
    """
    Calculates weighted binary cross entropy. The weights are determined dynamically
    by the balance of each category. This weight is calculated for each batch.

    The weights are calculted by determining the number of 'pos' and 'neg' classes
    in the true labels, then dividing by the number of total predictions.

    For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
    These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
    1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.

    This can be useful for unbalanced catagories.
    """
    # get the total number of inputs
    true = float(true)
    num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)

    # get weight of values in 'pos' category
    zero_weight = keras.backend.sum(true) / num_pred + keras.backend.epsilon()

    # get weight of values in 'false' category
    one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred + keras.backend.epsilon()

    # calculate the weight vector
    weights = (1.0 - true) * float(zero_weight) + true * one_weight

    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    # apply the weights
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)

# ---------------------------------------------------------------------------------------------------------------------
def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.

    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
        epsilon: Used for numerical stability to avoid divide by zero errors
    """

    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * np.sum(y_pred * y_true, axes)
    denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)

    return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch

# ---------------------------------------------------------------------------------------------------------------------
def TverskyLoss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    class_num = 2
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        TP = K.sum(y_true_f * y_pred_f)
        FP = K.sum((1 - y_true_f) * y_pred_f)
        FN = K.sum(y_true_f * (1 - y_pred_f))
        loss = ((TP + smooth) / (TP + alpha * FP + beta * FN + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    #
    # # flatten label and prediction tensors
    # inputs = K.flatten(inputs)
    # targets = K.flatten(targets)
    #
    # # True Positives, False Positives & False Negatives
    # TP = K.sum((inputs * targets))
    # FP = K.sum(((1 - targets) * inputs))
    # FN = K.sum((targets * (1 - inputs)))
    #
    # Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - total_loss