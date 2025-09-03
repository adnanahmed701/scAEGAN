import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from utils.image_pool import ImagePool
from models.discriminator import Discriminator
from models.generator import Generator
from models.data_loader import (load_data,
                                minibatchAB)
from models.networks_utils import (get_generator_function,
                                   get_generator_outputs)
from models.train_function import (generator_train_function,
                                   discriminator_A_train_function,
                                   discriminator_B_train_function,
                                   clip_weights)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


#config = tf.ConfigProto(
#device_count = {'GPU': 0}
# )
#sess = tf.Session(config=config)
#set_session(sess)


def create_networks(network_type, generator_params, discriminator_params):
    netG_A, real_A, cond_A, fake_B = Generator(network_type=network_type, name_suffix="_A", **generator_params)
    netG_B, real_B, cond_B, fake_A = Generator(network_type=network_type, name_suffix="_B", **generator_params)


    netD_A = Discriminator(network_type=network_type, **discriminator_params)
    netD_B = Discriminator(network_type=network_type, **discriminator_params)

    discriminators = (netD_A, netD_B)
    generators = (netG_A, netG_B)
    real_imgs = (real_A, real_B, cond_A, cond_B)
    fake_imgs = (fake_A, fake_B)

    return discriminators, generators, real_imgs, fake_imgs


def create_generator_functions(generators):
    netG_A, netG_B = generators

    netG_A_function = get_generator_function(netG_A)
    netG_B_function = get_generator_function(netG_B)

    return netG_A_function, netG_B_function


def create_train_functions(discriminators,
                           generators,
                           real_imgs,
                           fake_imgs,
                           loss_weights,
                           input_shape,
                           use_wgan):
    netG_train_function = generator_train_function(discriminators,
                                                           generators,
                                                           real_imgs,
                                                           fake_imgs,
                                                           loss_weights,
                                                           use_wgan)

    netD_A_train_function = discriminator_A_train_function(discriminators,
                                                                   generators,
                                                                   real_imgs,
                                                                   input_shape,
                                                                   use_wgan)

    netD_B_train_function = discriminator_B_train_function(discriminators,
                                                                   generators,
                                                                   real_imgs,
                                                                   input_shape,
                                                                   use_wgan)

    return netG_train_function, netD_A_train_function, netD_B_train_function


def create_image_pools(data_pool_size):
    fake_A_pool = ImagePool(pool_size=data_pool_size)
    fake_B_pool = ImagePool(pool_size=data_pool_size)

    return fake_A_pool, fake_B_pool


def create_batch_generators(data_path, train_file, test_file, input_shape, batch_size):
    train_A, label_A, ids_A = load_data(os.path.join(data_path, train_file[0]), input_shape)
    train_B, label_B, ids_B = load_data(os.path.join(data_path, train_file[1]), input_shape)

    train_batch = minibatchAB(train_A, label_A, train_B, label_B, batch_size=batch_size)

    test_A, label_A_test, ids_A_test = load_data(os.path.join(data_path, test_file[0]), input_shape)
    test_B, label_B_test, ids_B_test = load_data(os.path.join(data_path, test_file[1]), input_shape)
    test_batch = minibatchAB(test_A, label_A_test, test_B, label_B_test, batch_size=batch_size)

    batches_tuple = train_batch, test_batch
    test_data_tuple = test_A, test_B

    return (train_batch, test_batch), (test_A, test_B, label_A_test, label_B_test, ids_A_test, ids_B_test)


def save_networks(discriminators, generators, save_path):
    netD_A, netD_B = discriminators
    netG_A, netG_B = generators

    netG_A.save_weights(os.path.join(save_path, 'Generator_A_weights.h5'))
    netG_B.save_weights(os.path.join(save_path, 'Generator_B_weights.h5'))

    netD_A.save_weights(os.path.join(save_path, 'Discriminator_A_weights.h5'))
    netD_B.save_weights(os.path.join(save_path, 'Discriminator_B_weights.h5'))


def save_train_functions(train_functions, save_path):
    netG_train_function, netD_A_train_function, netD_B_train_function = train_functions

    netG_train_function.save_weights(os.path.join(save_path, 'Generator_train_function_weights.h5'))
    netD_A_train_function.save_weights(os.path.join(save_path, 'Discriminator_A_train_function_weights.h5'))
    netD_B_train_function.save_weights(os.path.join(save_path, 'Discriminator_B_train_function_weights.h5'))


def run_train_loop(train_settings,
                   train_functions,
                   generator_functions,
                   image_pools,
                   batches,
                   discriminators):
    batch_size, how_many_epochs, d_iters, discriminator_patience, use_data_pooling, use_wgan, print_cost = \
        train_settings
    netG_train_function, netD_A_train_function, netD_B_train_function = train_functions
    netG_A_function, netG_B_function = generator_functions
    fake_A_pool, fake_B_pool = image_pools
    train_batch, test_batch = batches
    netD_A, netD_B = discriminators

    time_start = time.time()
    iteration_count = 0
    epoch_count = 0
    display_freq = 500 // batch_size

    K.set_learning_phase(1)

    while epoch_count < how_many_epochs:
        target_label = np.zeros((batch_size, 1))
        epoch_count, A, B, label_A, label_B = next(train_batch)

        tmp_fake_B = netG_A_function([A, label_A])[0]
        tmp_fake_A = netG_B_function([B, label_B])[0]

        if use_data_pooling:
            _fake_B = fake_B_pool.query_over_images(tmp_fake_B)
            _fake_A = fake_A_pool.query_over_images(tmp_fake_A)
        else:
            _fake_B = tmp_fake_B
            _fake_A = tmp_fake_A

        if use_wgan:
            netD_B_train_function.train_on_batch([B, _fake_B, label_B], target_label)
            netD_A_train_function.train_on_batch([A, _fake_A, label_A], target_label)
            clip_weights(netD_B)
            clip_weights(netD_A)

            if iteration_count % d_iters == 0:
                netG_train_function.train_on_batch([A, B, label_A, label_B], target_label)
        else:
            netG_train_function.train_on_batch([A, B, label_A, label_B], target_label)

            if iteration_count % discriminator_patience == 0:
                netD_B_train_function.train_on_batch([B, _fake_B, label_B], target_label)
                netD_A_train_function.train_on_batch([A, _fake_A, label_A], target_label)


        iteration_count += 1

        if print_cost and iteration_count % display_freq == 0:
            target_label = np.zeros((batch_size, 1))

            epoch_count, A, B, label_A, label_B = next(test_batch)

            _fake_B = netG_A_function([A, label_A])[0]
            _fake_A = netG_B_function([B, label_B])[0]



            timecost = (time.time() - time_start) / 30
            print('\nEpoch_count: {}  iter_count: {}  timecost: {}mins'.format(epoch_count,
                                                                               iteration_count,
                                                                               timecost))
            print('\nDiscriminator A loss:', netD_A_train_function.evaluate([A, _fake_A, label_A], target_label))
            print('Discriminator B loss:', netD_B_train_function.evaluate([B, _fake_B, label_B], target_label))
            print('Generator loss:', netG_train_function.evaluate([A, B, label_A, label_B], target_label))


def _jitter(X, sigma=0.05, per_dim_scale=True, rng=None):
    rng = np.random.default_rng(rng)
    if per_dim_scale:
        # scale sigma by per-dimension std (avoid zero std)
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
        noise = rng.normal(0.0, sigma * std, size=X.shape)
    else:
        noise = rng.normal(0.0, sigma, size=X.shape)
    return X + noise

def _interp_classwise(X, y_onehot, k=5, alpha=0.5, rng=None):
    """
    Interpolate each point with a random neighbor from the SAME class (preserves conditional label).
    y_onehot: shape (n, C), one-hot.
    """
    rng = np.random.default_rng(rng)
    n, d = X.shape
    y = np.argmax(y_onehot, axis=1)
    X_new = X.copy()
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        if len(idx) < 2:
            continue
        nbr_idx = rng.choice(idx, size=len(idx))
        X_new[idx] = alpha * X[idx] + (1 - alpha) * X[nbr_idx]
    return X_new

def _upsample_latents(X, Y, ids, target_n, mode="jitter", sigma=0.05, alpha=0.5, k=5, seed=42):
    """
    X: (n,d) latent inputs; Y: (n,C) one-hot labels; ids: (n,) strings/ints
    Returns upsampled (X_out, Y_out, ids_out) with exactly target_n rows.
    """
    n = X.shape[0]
    if target_n <= n:
        # downsample deterministically for reproducibility
        return X[:target_n], Y[:target_n], ids[:target_n]

    reps = target_n // n
    rem  = target_n % n

    X_out = np.vstack([X for _ in range(reps)]) if reps else X[:0]
    Y_out = np.vstack([Y for _ in range(reps)]) if reps else Y[:0]
    ids_out = np.concatenate([ids for _ in range(reps)]) if reps else ids[:0]

    if rem:
        X_out = np.vstack([X_out, X[:rem]])
        Y_out = np.vstack([Y_out, Y[:rem]])
        ids_out = np.concatenate([ids_out, ids[:rem]])

    # augment features only (not labels)
    if mode == "jitter":
        X_out = _jitter(X_out, sigma=sigma, per_dim_scale=True, rng=seed)
    elif mode == "interp":
        X_out = _interp_classwise(X_out, Y_out, k=k, alpha=alpha, rng=seed)
    elif mode == "none":
        pass
    else:
        raise ValueError("gen_mode must be jitter|interp|none")

    # make new ids for synthetic rows to avoid collisions (optional)
    # append suffix for rows after the first n
    if target_n > n:
        base = np.asarray(ids_out, dtype=object)
        for i in range(n, target_n):
            base[i] = f"{base[i]}_aug{i-n+1}"
        ids_out = base

    return X_out, Y_out, ids_out



def process_test_data(generators_tuple, test_data_tuple, save_path,
                      num_generate=0, gen_mode="jitter", sigma=0.05, alpha=0.5, k=5, seed=42):
    netG_A, netG_B = generators_tuple
    test_A, test_B, label_A_test, label_B_test, ids_A_test, ids_B_test = test_data_tuple

    # ---- A note on shapes ----
    # test_A/test_B are AE latents (n,d), labels are one-hot (n,C), ids are 1D arrays.
    # If num_generate>0, upsample to that many rows; else, keep original size.

    # ===== Translate B -> A (uses netG_A with inputs [B, label_B])
    if num_generate and num_generate > 0:
        B_in, yB_in, idsB_in = _upsample_latents(test_B, label_B_test, ids_B_test,
                                                 target_n=num_generate, mode=gen_mode,
                                                 sigma=sigma, alpha=alpha, k=k, seed=seed)
    else:
        B_in, yB_in, idsB_in = test_B, label_B_test, ids_B_test

    fakeA, recB = get_generator_outputs(netG_B, netG_A, B_in, yB_in)  # returns (fake, recon)
    dfA = pd.DataFrame(fakeA)
    dfA.insert(0, 'case_id', idsB_in)
    dfA['condition'] = np.argmax(yB_in, axis=1).astype(int)
    dfA.to_csv(os.path.join(save_path, 'outdataB_OS.csv'), index=False)

    # ===== Translate A -> B (uses netG_B with inputs [A, label_A])
    if num_generate and num_generate > 0:
        A_in, yA_in, idsA_in = _upsample_latents(test_A, label_A_test, ids_A_test,
                                                 target_n=num_generate, mode=gen_mode,
                                                 sigma=sigma, alpha=alpha, k=k, seed=seed)
    else:
        A_in, yA_in, idsA_in = test_A, label_A_test, ids_A_test

    fakeB, recA = get_generator_outputs(netG_A, netG_B, A_in, yA_in)
    dfB = pd.DataFrame(fakeB)
    dfB.insert(0, 'case_id', idsA_in)
    dfB['condition'] = np.argmax(yA_in, axis=1).astype(int)
    dfB.to_csv(os.path.join(save_path, 'outdataA_OS.csv'), index=False)



def get_networks_params(input_shape, use_dropout, use_batch_norm, use_leaky_relu, use_wgan):
    generator_params = {
        'input_shape': input_shape,
        'use_dropout': use_dropout,
        'use_batch_norm': use_batch_norm,
        'use_leaky_relu': use_leaky_relu,
    }

    discriminator_params = {
        'input_shape': input_shape,
        'use_wgan': use_wgan,
        'use_batch_norm': use_batch_norm,
        'use_leaky_relu': use_leaky_relu,
    }

    return generator_params, discriminator_params


def train_model(network_parameters,
                loss_weights,
                train_settings,
                batches,
                test_data,
                generator_params,
                discriminator_params,
                saving,
                num_generate,
                gen_mode,
                sigma,
                alpha,
                k,
                seed):

    network_type, input_shape, use_wgan, data_pool_size = network_parameters
    save_path, save_model = saving

    K.set_learning_phase(1)

    discriminators, generators, real_imgs, fake_imgs = \
        create_networks(network_type, generator_params, discriminator_params)

    train_functions = \
        create_train_functions(discriminators,
                               generators,
                               real_imgs,
                               fake_imgs,
                               loss_weights,
                               input_shape,
                               use_wgan)

    generator_functions = create_generator_functions(generators)

    image_pools = create_image_pools(data_pool_size)

    run_train_loop(train_settings,
                   train_functions,
                   generator_functions,
                   image_pools,
                   batches,
                   discriminators)

    process_test_data(
    generators, test_data, save_path,
    num_generate=num_generate,
    gen_mode=gen_mode,
    sigma=sigma,
    alpha=alpha,
    k=k,
    seed=seed
)


    if save_model:
        save_networks(discriminators, generators, save_path)
        save_train_functions(train_functions, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="cGAN")
    parser.add_argument("--input_shape", default=50, type=int)

    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--d_iters", default=5, type=int)
    parser.add_argument("--discriminator_patience", default=1, type=int)

    parser.add_argument("--use_wgan", default=True, type=bool)
    parser.add_argument("--use_batch_norm", default=True, type=bool)
    parser.add_argument("--use_leaky_relu", default=True, type=bool)
    parser.add_argument("--use_dropout", default=True, type=bool)
    parser.add_argument("--use_data_pooling", default=False, type=bool)

    parser.add_argument("--cycle_loss_weight", default=.3, type=float)
    parser.add_argument("--id_loss_weight", default=.3, type=float)
    parser.add_argument("--data_pool_size", default=500, type=int)

    parser.add_argument("--data_path", default="../inputdata/")
    parser.add_argument("--train_file", nargs=2, type=str, required=True)
    parser.add_argument("--test_file", nargs=2, type=str, required=True)

    parser.add_argument("--save_path", default="../savepath/")
    parser.add_argument("--save_model", default=True, type=bool)

    parser.add_argument("--print_cost", default=True, type=bool)
    parser.add_argument("--num_generate", type=int, default=0,
                        help="If >0, upsample test inputs to this many rows per domain before translation.")
    parser.add_argument("--gen_mode", default="jitter", choices=["jitter","interp","none"],
                        help="How to synthesize extra latent rows.")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="Std for jitter (can be scaled per-dimension).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Blend factor for interpolation (0..1).")
    parser.add_argument("--k", type=int, default=5,
                        help="Neighbors for interpolation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")


    args = parser.parse_args()
    
    input_shape = (args.input_shape,)

    network_parameters = (args.network_type, input_shape, args.use_wgan, args.data_pool_size)
    saving = (args.save_path, args.save_model)

    loss_weights = (args.cycle_loss_weight, args.id_loss_weight)

    train_settings = (args.batch_size, args.epochs, args.d_iters,
                            args.discriminator_patience, args.use_data_pooling,
                            args.use_wgan, args.print_cost)

    batches, test_data = \
        create_batch_generators(args.data_path, args.train_file, args.test_file,
                                input_shape, args.batch_size)

    generator_params, discriminator_params = \
        get_networks_params(input_shape, args.use_dropout, args.use_batch_norm,
                            args.use_leaky_relu, args.use_wgan)

    train_model(network_parameters,
            loss_weights,
            train_settings,
            batches,
            test_data,
            generator_params,
            discriminator_params,
            saving,
            args.num_generate,
            args.gen_mode,
            args.sigma,
            args.alpha,
            args.k,
            args.seed)


if __name__ == "__main__":
    main()
