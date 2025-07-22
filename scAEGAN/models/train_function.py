from itertools import chain
import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (Input, BatchNormalization, Lambda, Concatenate)

from .loss import generator_loss, discriminator_loss

# Add this at the top or pass it explicitly if variable
condition_dim = 10

def get_train_function(inputs, loss_function, lambda_layer_inputs):
    adam = Adam(lr=0.0005, beta_1=0.5, beta_2=0.999, epsilon=1e-7, decay=0)
    train_function = Model(inputs, Lambda(loss_function)(lambda_layer_inputs))
    train_function.compile(adam, 'mse')
    return train_function

def generator_train_function(discriminators, generators, real_imgs, fake_imgs, loss_weights, use_wgan=True):
    netD_A, netD_B = discriminators
    netG_A, netG_B = generators
    real_A, real_B, label_A, label_B = real_imgs
    fake_A, fake_B = fake_imgs
    cycle_loss_weight, id_loss_weight = loss_weights

    netD_B_predict_fake = netD_B([fake_B, label_B])
    rec_A = netG_B([fake_B, label_B])

    netD_A_predict_fake = netD_A([fake_A, label_A])
    rec_B = netG_A([fake_A, label_A])

    lambda_layer_inputs = [netD_B_predict_fake, rec_A, real_A, netD_A_predict_fake, rec_B, real_B, fake_A, fake_B]

    for layer in chain(netG_A.layers, netG_B.layers):
        layer.trainable = True

    for layer in chain(netD_A.layers, netD_B.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netG_loss_partial = lambda x: generator_loss(x,
                                                 cycle_loss_weight=cycle_loss_weight,
                                                 id_loss_weight=id_loss_weight,
                                                 use_wgan=use_wgan)

    netG_train_function = get_train_function(inputs=[real_A, real_B, label_A, label_B],
                                             loss_function=netG_loss_partial,
                                             lambda_layer_inputs=lambda_layer_inputs)
    return netG_train_function


def discriminator_A_train_function(discriminators, generators, real_imgs, input_shape, use_wgan=False):
    netD_A, netD_B = discriminators
    netG_A, netG_B = generators
    real_A, _, label_A, _ = real_imgs

    _fake_A = Input(shape=input_shape, name="fake_A")
    label_input_A = Input(shape=(condition_dim,), name="label_A")

    netD_A_predict_real = netD_A([real_A, label_input_A])
    netD_A_predict_fake = netD_A([_fake_A, label_input_A])

    for l in netD_A.layers:
        l.trainable = True

    for layer in chain(netG_A.layers, netG_B.layers, netD_B.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netD_loss_partial = lambda x: discriminator_loss(x, use_wgan=use_wgan)
    netD_A_train_function = get_train_function(inputs=[real_A, _fake_A, label_input_A],
                                               loss_function=netD_loss_partial,
                                               lambda_layer_inputs=[netD_A_predict_real,
                                                                    netD_A_predict_fake])
    return netD_A_train_function


def discriminator_B_train_function(discriminators, generators, real_imgs, input_shape, use_wgan=False):
    netD_A, netD_B = discriminators
    netG_A, netG_B = generators
    _, real_B, _, label_B = real_imgs

    _fake_B = Input(shape=input_shape, name="fake_B")
    label_input_B = Input(shape=(condition_dim,), name="label_B")

    netD_B_predict_real = netD_B([real_B, label_input_B])
    netD_B_predict_fake = netD_B([_fake_B, label_input_B])

    for l in netD_B.layers:
        l.trainable = True

    for layer in chain(netG_A.layers, netG_B.layers, netD_A.layers):
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer._per_input_updates = {}

    netD_loss_partial = lambda x: discriminator_loss(x, use_wgan=use_wgan)
    netD_B_train_function = get_train_function(inputs=[real_B, _fake_B, label_input_B],
                                               loss_function=netD_loss_partial,
                                               lambda_layer_inputs=[netD_B_predict_real,
                                                                    netD_B_predict_fake])
    return netD_B_train_function


def clip_weights(net, clip_lambda=.1):
    weights = [np.clip(w, -clip_lambda, clip_lambda) for w in net.get_weights()]
    net.set_weights(weights)
