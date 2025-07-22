from keras import backend as K
from keras.layers import (Activation,
                          Add,
                          BatchNormalization,
                          Dropout,
                          LeakyReLU)
from keras.layers.core import Dense


def activation_layer(use_leaky_relu=True, leaky_alpha=0.1):
    if use_leaky_relu:
        return LeakyReLU(alpha=leaky_alpha)
    else:
        return Activation('relu')


def dense_layer(x, units=50, use_batch_norm=True, use_leaky_relu=False):
    x = Dense(units=units, activation=None)(x)

    if use_batch_norm:
        x = BatchNormalization()(x)

    return activation_layer(use_leaky_relu)(x)


from keras.layers import Lambda

def residual_dense_block(x, units=50, use_dropout=False, use_batch_norm=True, use_leaky_relu=False):
    input_dim = K.int_shape(x)[-1]

    y = dense_layer(x, units=units, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    if use_dropout:
        y = Dropout(0.2)(y)

    y = Dense(units=units, activation=None)(y)

    if use_batch_norm:
        y = BatchNormalization()(y)

    # 🔧 Project x to match units if needed
    if input_dim != units:
        x = Dense(units=units, activation=None)(x)

    y = Add()([y, x])
    y = activation_layer(use_leaky_relu)(y)

    return y



def get_generator_function(netG):
    real_input = netG.inputs[0]
    label_input = netG.inputs[1]
    fake_output = netG.outputs[0]
    function = K.function([real_input, label_input], [fake_output])
    return function


def get_generator_outputs(netG_alpha, netG_beta, real_input, label_input):
    fake_output = netG_alpha.predict([real_input, label_input])
    rec_input = netG_beta.predict([fake_output, label_input])
    outputs = [fake_output, rec_input]
    return outputs
