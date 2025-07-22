from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from .networks_utils import dense_layer

def Discriminator_cGAN(input_shape=(50,), condition_dim=10, use_wgan=False, use_batch_norm=True, use_leaky_relu=False):
    input_data = Input(shape=input_shape, name="data_input")
    condition = Input(shape=(condition_dim,), name="condition_input")

    # Combine inputs
    x = Concatenate()([input_data, condition])

    # Dense layers
    x = dense_layer(x, units=30, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)
    x = dense_layer(x, units=50, use_batch_norm=use_batch_norm, use_leaky_relu=use_leaky_relu)

    # Output layer
    activation = None if use_wgan else "sigmoid"
    outputs = Dense(units=1, activation=activation)(x)

    return Model(inputs=[input_data, condition], outputs=outputs)


def Discriminator(network_type='cGAN', **args):
    assert network_type in {'cGAN'}, "Network 'network_type'!!!"

    generators = {
        "cGAN": Discriminator_cGAN,
    }

    return generators[network_type](**args)
