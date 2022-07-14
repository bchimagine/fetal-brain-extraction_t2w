import numpy as np
from keras import backend as K
from keras.layers import (
    Activation,
    Add,
    Concatenate,
    Input,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    BatchNormalization,
    Lambda,
    DepthwiseConv2D,
    Conv2DTranspose,
    Dropout,
    SeparableConv2D,
)
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.optimizers import Adam


# from .metrics import mean_iou


def channel_shuffle(x, groups):
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups

    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))
    x = K.reshape(x, [-1, height, width, in_channels])

    return x


def group_conv(x, in_channels, out_channels, groups, kernel=1, stride=1, name=''):
    if groups == 1:
        return Conv2D(filters=out_channels, kernel_size=kernel, padding='same',
                      use_bias=False, strides=stride, name=name)(x)

    ig = in_channels // groups
    group_list = []

    assert out_channels % groups == 0

    for i in range(groups):
        offset = i * ig
        group = Lambda(lambda z: z[:, :, :, offset: offset + ig], name='%s/g%d_slice' % (name, i))(x)
        group_list.append(Conv2D(int(0.5 + out_channels / groups), kernel_size=kernel, strides=stride,
                                 use_bias=False, padding='same', name='%s_/g%d' % (name, i))(group))
    return Concatenate(name='%s/concat' % name)(group_list)


def shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides=2, stage=1, block=1):
    prefix = 'stage{}/block{}'.format(stage, block)
    bn_axis = 3

    bottleneck_channels = int(out_channels * bottleneck_ratio)
    groups = (1 if stage == 2 and block == 1 else groups)

    x = group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                   groups=(1 if stage == 2 and block == 1 else groups),
                   name='%s/1x1_gconv_1' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_1' % prefix)(x)
    x = Activation('relu', name='%s/relu_gconv_1' % prefix)(x)

    x = Lambda(channel_shuffle, arguments={'groups': groups}, name='%s/channel_shuffle' % prefix)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False,
                        strides=strides, name='%s/1x1_dwconv_1' % prefix)(x)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_dwconv_1' % prefix)(x)

    x = group_conv(x, bottleneck_channels,
                   out_channels=out_channels if strides == 1 else out_channels - in_channels,
                   groups=groups, name='%s/1x1_gconv_2' % prefix)
    x = BatchNormalization(axis=bn_axis, name='%s/bn_gconv_2' % prefix)(x)

    if strides < 2:
        x = Add(name='%s/add' % prefix)([x, inputs])
    else:
        avg = AveragePooling2D(pool_size=3, strides=2, padding='same', name='%s/avg_pool' % prefix)(inputs)
        x = Concatenate(bn_axis, name='%s/concat' % prefix)([x, avg])

    x = Activation('relu', name='%s/relu_out' % prefix)(x)
    return x


def stage(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    x = shuffle_unit(x, in_channels=channel_map[stage - 2],
                     out_channels=channel_map[stage - 1], strides=2,
                     groups=groups, bottleneck_ratio=bottleneck_ratio,
                     stage=stage, block=1)

    for i in range(1, repeat + 1):
        x = shuffle_unit(x, in_channels=channel_map[stage - 1],
                         out_channels=channel_map[stage - 1], strides=1,
                         groups=groups, bottleneck_ratio=bottleneck_ratio,
                         stage=stage, block=(i + 1))

    return x

def stage_decoder(x, channel_map, bottleneck_ratio, repeat=1, groups=1, stage=1):
    for i in range(1, repeat + 1):
        x = shuffle_unit(x, in_channels=channel_map[stage - 1],
                         out_channels=channel_map[stage - 1], strides=1,
                         groups=groups, bottleneck_ratio=bottleneck_ratio,
                         stage=stage, block=(i + 1))
    return x


def shufflenet(inp, output_channels):
    shuffle_units = [3, 7, 3]
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    groups = 2
    scale_factor = 1.0
    bottleneck_ratio = 0.25

    exp = np.insert(np.arange(0, len(shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    out_channels_in_stage = np.array([32, 64, 128, 256]).astype(int)
    # ShuffleNet encoder
    x = Conv2D(out_channels_in_stage[0], 3, strides=1, padding='same', use_bias=False, activation='relu', name='conv1')(inp)
    feed0 = x
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)



    repeat = shuffle_units[0]
    x = stage(x, out_channels_in_stage, bottleneck_ratio,
              stage=2, groups=groups, repeat=repeat)
    feed2 = x

    repeat = shuffle_units[1]
    x = stage(x, out_channels_in_stage, bottleneck_ratio,
              stage=3, groups=groups, repeat=repeat)
    feed1 = x

    repeat = shuffle_units[2]
    x = stage(x, out_channels_in_stage, bottleneck_ratio,
              stage=4, groups=groups, repeat=repeat)

    # scorefr
    #x = Conv2D(output_channels, (1, 1), padding='same', name='conv_1c_1x1')(x)
    scorefr = x

    return feed0, feed2, feed1, scorefr

def dechannel(inputs, in_channels, n_filter):
    bn_axis = 3

    bottleneck_channels = int(n_filter * 0.25)
    groups = 1

    x = group_conv(inputs, in_channels, out_channels=bottleneck_channels,
                   groups=groups)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)

    x = Lambda(channel_shuffle, arguments={'groups': groups})(x)
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False,
                        strides=1)(x)
    x = BatchNormalization(axis=bn_axis)(x)

    x = group_conv(x, bottleneck_channels,
                   out_channels=n_filter,
                   groups=groups)
    x = BatchNormalization(axis=bn_axis)(x)


    ##
    y = DepthwiseConv2D(kernel_size=(3, 3), padding='same', use_bias=False,
                        strides=1)(inputs)
    y = BatchNormalization(axis=bn_axis)(y)

    y = group_conv(y, in_channels,
                   out_channels=n_filter,
                   groups=groups)
    y = BatchNormalization(axis=bn_axis)(y)


    x = Add()([x, y])

    x = Activation('relu')(x)
    return x


def conv2d_block(input_tensor, n_filters, kernel_size=3):
    '''
    Adds 2 convolutional layers with the parameters passed to it

    Args:
      input_tensor (tensor) -- the input tensor
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size for the convolution

    Returns:
      tensor of output features
    '''
    # first layer
    # x = input_tensor
    # #x = dechannel(x, in_channels, n_filters)
    # #x = dechannel(x, n_filters, n_filters)
    # #for i in range(2):
    # x = Conv2D(filters=n_filters/2, kernel_size=(1, 1),
    #           kernel_initializer='he_normal', padding='same')(x)
    # x = tf.keras.layers.Activation('relu')(x)
    # x = Conv2D(filters=n_filters, kernel_size=(1, 1),
    #           kernel_initializer='he_normal', padding='same')(x)
    # x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
    #            kernel_initializer='he_normal', padding='same')(x)
    # x = tf.keras.layers.Activation('relu')(x)
    x = input_tensor
    for i in range(1):
        x = tf.keras.layers.SeparableConv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3, bottleneck_ratio=0.25,
                  stage=3, groups=1, repeat=1):
    '''
    defines the one decoder block of the UNet

    Args:
      inputs (tensor) -- batch of input features
      conv_output (tensor) -- features from an encoder block
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size
      strides (int) -- strides for the deconvolution/upsampling
      padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

    Returns:
      c (tensor) -- output features of the decoder block
    '''
    U = tf.keras.layers.Conv2D(n_filters[stage-5], 1, strides=1, padding='same')(inputs)
    u = tf.keras.layers.Conv2DTranspose(n_filters[stage-5], kernel_size, strides=strides, padding='same')(U)
    #u = Conv2DTranspose(n_filters[stage-5], kernel_size, strides=strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = Dropout(dropout)(c)

    c = conv2d_block(c, n_filters[stage - 5], kernel_size=3)

    return c


def decoder(buttleneck, feed0, feed2, feed1, scorefr, output_channels):
    shuffle_units = [2, 2, 2]
    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    groups = 2
    scale_factor = 1.0
    bottleneck_ratio = 0.25

    exp = np.insert(np.arange(0, len(shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)
    out_channels_in_stage = np.array([32, 64, 128, 256]).astype(int)


    repeat = shuffle_units[0]
    x = decoder_block(buttleneck, scorefr, n_filters=out_channels_in_stage, kernel_size=(4, 4), strides=(2, 2), dropout=0.3, bottleneck_ratio=0.25,
                  stage=4, groups=groups, repeat=repeat)
    
    x = decoder_block(x, feed1, n_filters=out_channels_in_stage, kernel_size=(4, 4), strides=(2, 2), dropout=0.3, bottleneck_ratio=0.25,
                  stage=3, groups=groups, repeat=repeat)
    
    x = decoder_block(x, feed2, n_filters=out_channels_in_stage, kernel_size=(4, 4), strides=(2, 2), dropout=0.3, bottleneck_ratio=0.25,
                  stage=2, groups=groups, repeat=repeat)

    x = decoder_block(x, feed0, n_filters=out_channels_in_stage, kernel_size=(8, 8), strides=(4, 4), dropout=0.3,
                      bottleneck_ratio=0.25,
                      stage=1, groups=groups, repeat=repeat)


    outputs = Conv2D(output_channels, (1, 1), activation='softmax')(x)

    return outputs


def shufflenet_unet(input_shape, cls_num=2):
    inp = Input(input_shape)
    feed0, feed2, feed1, scorefr = shufflenet(inp, cls_num)

    buttleneck = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2),
                                   kernel_initializer='he_normal', padding='same')(scorefr)
    buttleneck= Activation('relu')(buttleneck)
    # decoder
    decoder_ = decoder(buttleneck, feed0, feed2, feed1, scorefr, cls_num)

    return Model(inp, decoder_, name='ShuffleSeg_v2')