import tensorflow as tf


# Encoder Utilities
def conv2d_block(input_tensor, n_filters, kernel_size=3):

    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3):

    f = conv2d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(f)
    p = tf.keras.layers.Dropout(0.3)(p)

    return f, p


def encoder(inputs):

    f1, p1 = encoder_block(inputs, n_filters=32, pool_size=(2, 2), dropout=0.3)
    f2, p2 = encoder_block(p1, n_filters=64, pool_size=(2, 2), dropout=0.3)
    f3, p3 = encoder_block(p2, n_filters=128, pool_size=(2, 2), dropout=0.3)
    # f4, p4 = encoder_block(p3, n_filters=256, pool_size=(2, 2), dropout=0.3)

    # return p4, (f1, f2, f3, f4)
    return p3, (f1, f2, f3)


def bottleneck(inputs):

    bottle_neck = conv2d_block(inputs, n_filters=256)

    return bottle_neck


def decoder_block(inputs, conv_output, n_filters=64, kernel_size=3, strides=3, dropout=0.3):

    u = tf.keras.layers.Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv2d_block(c, n_filters, kernel_size=3)

    return c


def decoder(inputs, convs, output_channels):

    f1, f2, f3 = convs

    # c6 = decoder_block(inputs, f4, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c7 = decoder_block(inputs, f3, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c8 = decoder_block(c7, f2, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    c9 = decoder_block(c8, f1, n_filters=32, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)

    # outputs = tf.keras.layers.Conv2D(output_channels, (1, 1), activation='softmax')(c9)

    return c9



def initial_block(inp, nb_filter=15, nb_row=3, nb_col=3, strides=(2, 2)):
    conv = tf.keras.layers.Conv2D(nb_filter, (nb_row, nb_col), padding='same', strides=strides)(inp)
    conv = tf.keras.layers.Activation('relu')(conv)
    max_pool = tf.keras.layers.MaxPooling2D()(inp)
    merged = tf.keras.layers.concatenate([conv, max_pool], axis=3)
    return merged


def twopath(input_shape, cls_num=2):

    # specify the input shape
    inputs = tf.keras.layers.Input(shape=input_shape)

    init = initial_block(inputs)
    # init = tf.keras.layers.Activation('relu')(init)
    # feed the inputs to the encoder
    encoder_output, convs = encoder(init)

    # feed the encoder output to the bottleneck
    bottle_neck = bottleneck(encoder_output)

    # feed the bottleneck and encoder block outputs to the decoder
    # specify the number of classes via the `output_channels` argument
    dec = decoder(bottle_neck, convs, output_channels=cls_num)

    init_ = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    init_ = tf.keras.layers.Activation('relu')(init_)
    init_ = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(init_)
    init_ = tf.keras.layers.Activation('relu')(init_)

    out = tf.keras.layers.Add()([init_, dec])
    outputs = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(out)
    # outputs = conv2d_block(outputs, n_filters=32)
    # outputs = tf.keras.layers.Dropout(0.3)(outputs)
    # outputs = conv2d_block(outputs, 2, kernel_size=3)

    outputs = tf.keras.layers.Conv2D(cls_num, (1, 1), activation='softmax')(outputs)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

if __name__ == '__main__':
    from keras_flops import get_flops

    model = twopath(input_shape=(256, 256, 1), cls_num=2)
    model.summary()

    flops = get_flops(model, batch_size=1)
    print(f"FLOPS: {flops / 10 ** 9:.03} G")