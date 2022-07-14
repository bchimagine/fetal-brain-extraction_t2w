# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from train import train
import time
from error import *
from utilities import class_wise_metrics

# def exe():
#     ''' Read Data '''
X_train = np.load("../data/X_sel.npy")
y_train = np.load("../data/y_sel.npy")
y_train = tf.keras.utils.to_categorical(y_train, 2)

X_test_normal = np.load("../data/X_test_normal.npy")
y_test_normal = np.load("../data/y_test_normal.npy")

X_test_challenging = np.load("../data/X_test_challenging.npy")
y_test_challenging = np.load("../data/y_test_challenging.npy")

''' Normalize '''
X_train = np.divide(X_train.astype(float), np.std(X_train.astype(float), axis=0),
                    out=np.zeros_like(X_train.astype(float)),
                    where=np.std(X_train.astype(float), axis=0) != 0)

X_test_normal = np.divide(X_test_normal.astype(float), np.std(X_test_normal.astype(float), axis=0),
                          out=np.zeros_like(X_test_normal.astype(float)),
                          where=np.std(X_test_normal.astype(float), axis=0) != 0)
#
# X_test_challenging = np.divide(X_test_challenging.astype(float), np.std(X_test_challenging.astype(float), axis=0),
#                                out=np.zeros_like(X_test_challenging.astype(float)),
#                                where=np.std(X_test_challenging.astype(float), axis=0) != 0)

model_names = ['dfanet', 'enet', 'enet_unet', 'fastscnn', 'icnet', 'mobilenet', 'shufflenet_unet', 'shuffleseg',
               'shuffleseg_v2', 'unet', 'unet_v2']

out = np.empty((0, 3))
out_time = np.empty((0, 4))
for name in model_names:
    print(name)
    model = train(X_train, y_train, name)
    for bs in range(7):
        batch_size = 2**bs

        ''' Test Normal Data'''
        pred_time_list = []
        for i in range(10):
            start_time = time.time()
            mask = model.predict(X_test_normal, batch_size=batch_size)
            total_time = time.time() - start_time
            mean_time = total_time / mask.shape[0]
            pred_time_list.append(mean_time)

        duration_m = np.mean(pred_time_list[1:])
        duration_std = np.std(pred_time_list[1:])
        o_t = np.array([[name, batch_size, duration_m, duration_std]], dtype=object)
        out_time = np.vstack((out_time, o_t))

    results = np.argmax(mask, axis=3)
    results = results[..., tf.newaxis]
    # np.save('../results/results/'+name+'_mask.npy', results)

    # compute the class wise metrics
    cls_wise_iou, cls_wise_dice_score = class_wise_metrics(y_test_normal, results)
    mDice = (np.array(cls_wise_dice_score)).mean()
    mIoU = (np.array(cls_wise_iou)).mean()

    o = np.array([[name, mDice, mIoU]], dtype=object)
    out = np.vstack((out, o))

np.save('../misc/results/out_time.npy', out_time)
np.save('../misc/results/out.npy', out)


# return result




# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     exe()
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
