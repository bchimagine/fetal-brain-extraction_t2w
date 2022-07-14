import os
import warnings
from error import *

import tensorflow as tf
import keras

from models.resunet import ResUNet
from models.dfanet import dfanet
from models.enet import enet
from models.fastscnn import Fast_SCNN
from models.icnet import IcNet
from models.mobilenet import MobileNet
from models.shuffleseg import ShuffleSeg
from models.unet import unet
from models.segnet import segnet
from models.shuffleseg_v2 import ShuffleSeg_v2
from models.enet_unet import enet_unet
from models.shufflenet_unet import shufflenet_unet
from misc.unet_v3 import unet_v3
from models.unet_v4 import unet_v4
from misc.unet_v6 import unet_v6
from models.twopath import twopath


# ---------------------------------------------------------------------------------------------------------------------
class DataGenerator(keras.utils.all_utils.Sequence):
    def __init__(self, x_data, y_data, batch_size):
        self.x, self.y = x_data, y_data
        self.batch_size = batch_size
        self.num_batches = np.ceil(len(x_data) / batch_size)
        self.batch_idx = np.array_split(range(len(x_data)), self.num_batches)

    def __len__(self):
        return len(self.batch_idx)

    def __getitem__(self, idx):
        batch_x = self.x[self.batch_idx[idx]]
        batch_y = self.y[self.batch_idx[idx]]
        return batch_x, batch_y


# ---------------------------------------------------------------------------------------------------------------------
def get_model(name):
    if name == 'dfanet':
        model = dfanet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num, size_factor=2)
    elif name == 'enet':
        model = enet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'fastscnn':
        model = Fast_SCNN(num_classes=cls_num, input_shape=(IMG_SIZE, IMG_SIZE, cha_num)).model()
    elif name == 'icnet':
        model = IcNet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'mobilenet':
        model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shuffleseg':
        model = ShuffleSeg(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'unet':
        model = unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    #
    elif name == 'unet_v6':
        model = unet_v6(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'unet_v4':
        model = unet_v4(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'unet_v3':
        model = unet_v3(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shuffleseg_v2':
        model = ShuffleSeg_v2(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'enet_unet':
        model = enet_unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'shufflenet_unet':
        model = shufflenet_unet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=cls_num)
    elif name == 'segnet':
        model = segnet(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), n_labels=2)

    elif name == 'twopath':
        model = twopath(input_shape=(IMG_SIZE, IMG_SIZE, cha_num), cls_num=2)

    else:
        raise NameError("No Model Found!")
    return model


# ---------------------------------------------------------------------------------------------------------------------
''' Read Data '''
X_train = np.load("../data/X_train.npy", mmap_mode="r")
y_train = np.load("../data/y_train.npy", mmap_mode="r")
y_train = tf.keras.utils.to_categorical(y_train, 2)

''' Normalize '''

idx = np.any(X_train != 0, axis=(1, 2, 3))
X_train = X_train[idx]
y_train = y_train[idx]
X_train = np.divide(X_train.astype(float), np.std(X_train.astype(float), axis=0),
                    out=np.zeros_like(X_train.astype(float)),
                    where=np.std(X_train.astype(float), axis=0) != 0)

train_generator = DataGenerator(X_train[3866:], y_train[3866:], batch_size=16)
test_generator = DataGenerator(X_train[:3866], y_train[:3866], batch_size=16)

smooth = 1.

''' Initialization '''
IMG_SIZE = 256
cha_num = 1
cls_num = 2

model_name = 'unet'
model = get_model(model_name)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1.0e-4,
    decay_steps=2000,
    decay_rate=0.9)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=dice_coef_loss,
              metrics=[dice_coef])

''' Start Train '''
# start_time_train = time.time()
model_history = model.fit(train_generator, steps_per_epoch=int(15464 // 16),
                          validation_data=test_generator, validation_steps=int(3866 // 16),
                          epochs=100)
#
# model_history = model.fit(X_train, y_train,
#                           batch_size=8,
#                           epochs=150,
#                           validation_split=0.2)
# duration_train = (time.time() - start_time_train)
# print('train time %s seconds' % duration_train)

save_path = '../results/'
model.save(save_path + model_name + '.model')
model.save_weights(save_path + model_name + '_weights.h5')
