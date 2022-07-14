# based on shufflenetV1
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
)
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

    # ShuffleNet encoder
    x = Conv2D(24, 3, strides=(2, 2), padding='same', use_bias=False, activation='relu', name='conv1')(inp)
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
    x = Conv2D(output_channels, (1, 1), padding='same', name='conv_1c_1x1')(x)
    scorefr = x

    return feed2, feed1, scorefr


def fcn8s(feed2, feed1, scorefr, output_channels):
    # upscore2s
    upscore2 = Conv2DTranspose(output_channels, (4, 4), strides=(2, 2), padding='same', use_bias=False)(scorefr)
    score_feed1 = Conv2D(output_channels, (1, 1), use_bias=False)(feed1)
    fuse_feed1 = Add()([score_feed1, upscore2])

    # upscore4s
    upscore4 = Conv2DTranspose(output_channels, (4, 4), strides=(2, 2), padding='same', use_bias=False)(fuse_feed1)
    score_feed2 = Conv2D(output_channels, (1, 1), use_bias=False)(feed2)
    fuse_feed2 = Add()([score_feed2, upscore4])

    # upscore8s
    upscore8 = Conv2DTranspose(output_channels, (16, 16), strides=(8, 8), padding='same', use_bias=False, activation='softmax')(fuse_feed2)

    return upscore8

def ShuffleSeg(input_shape, cls_num=3):
    inp = Input(input_shape)
    feed2, feed1, scorefr = shufflenet(inp, cls_num)

    # decoder
    decoder = fcn8s(feed2, feed1, scorefr, cls_num)

    return Model(inp, decoder, name='ShuffleSeg')

if __name__ == '__main__':
    from keras_flops import get_flops

    model = ShuffleSeg(input_shape=(256, 256, 3), cls_num=3)
    model.summary()

    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")