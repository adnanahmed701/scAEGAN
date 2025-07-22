from keras.layers import Input, Concatenate, Dense
from keras.models import Model

from .networks_utils import residual_dense_block, dense_layer

def Generator_cGAN(input_shape=(50,), condition_dim=10, use_dropout=True, use_batch_norm=True,
                   use_leaky_relu=False, name_suffix=""):
    input_data = Input(shape=input_shape, name=f"data_input{name_suffix}")
    condition = Input(shape=(condition_dim,), name=f"condition_input{name_suffix}")

    # Combine inputs
    x = Concatenate(name=f"concat{name_suffix}")([input_data, condition])

    # Pass through residual dense block
    x = residual_dense_block(
        x,
        units=50,
        use_dropout=use_dropout,
        use_batch_norm=use_batch_norm,
        use_leaky_relu=use_leaky_relu
    )

    # Final output layer (same size as input_data)
    outputs = Dense(units=input_shape[0], activation=None, name=f"output{name_suffix}")(x)

    model = Model(inputs=[input_data, condition], outputs=outputs, name=f"Generator{name_suffix}")
    return model, input_data, condition, outputs


def Generator(network_type='cGAN', name_suffix="", **args):
    assert network_type in {'cGAN'}, "Unsupported network type. Only 'cGAN' is supported."

    generators = {
        "cGAN": Generator_cGAN,
    }

    return generators[network_type](name_suffix=name_suffix, **args)
