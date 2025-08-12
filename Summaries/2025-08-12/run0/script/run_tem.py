#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: James Whittington
"""

import data_utils
import model_utils
import parameters

import numpy as np
import tensorflow as tf

# # Force TensorFlow to use CPU only
# tf.config.set_visible_devices([], 'GPU')

import glob
import os
import shutil
import tem_model as tem
import time
import importlib
import argparse

importlib.reload(tem)

parser = argparse.ArgumentParser(description='Train TEM')
parser.add_argument('--fast', action='store_true', help='Use a smaller, low-memory configuration')
parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
parser.add_argument('--seq-len', type=int, default=None, help='Override sequence length')
parser.add_argument('--world', type=str, default=None, help='Override world type (e.g., rectangle)')
args, _ = parser.parse_known_args()

# Create directories for storing all information about the current run
run_path, train_path, model_path, save_path, script_path, envs_path = data_utils.make_directories()

# def print_var_info(name, var):
#     tname = type(var).__name__
#     if isinstance(var, (np.ndarray, tf.Tensor)):
#         print(f"{name}: type={tname}, shape={tuple(var.shape)}")
#     elif isinstance(var, str):
#         print(f"{name}: type={tname}, value={var}")
#     else:
#         print(f"{name}: type={tname}")

# print_var_info("run_path", run_path)
# print_var_info("train_path", train_path)
# print_var_info("model_path", model_path)
# print_var_info("save_path", save_path)
# print_var_info("script_path", script_path)
# print_var_info("envs_path", envs_path)

# Save all python files in current directory to script directory
files = glob.iglob(os.path.join('', '*.py'))
for file in files:
    if os.path.isfile(file):
        shutil.copy2(file, os.path.join(script_path, file))

# Initialise hyper-parameters for model
world_override = args.world if args.world else ('rectangle' if args.fast else None)
params = parameters.default_params(world_type=world_override, batch_size=args.batch_size)

if args.fast:
    # Smaller data/model settings for lower memory
    params.batch_size = params.batch_size if args.batch_size else 2
    params.n_envs = params.batch_size
    params.seq_len = args.seq_len if args.seq_len else 25
    params.tf_range = True  # tf.range uses less RAM
    params.use_reward = False

    # Reduce sensory and model sizes
    params.s_size_comp = 6
    # Recompute sensory alphabet size and two-hot table/matrix
    s_size_fast = len(parameters.combins_table(params.s_size_comp, 2))
    params.s_size = s_size_fast
    params.two_hot_mat = parameters.onehot2twohot(
        np.expand_dims(np.eye(params.s_size), axis=0),
        parameters.combins_table(params.s_size_comp, 2),
        params.s_size_comp,
    )

    # Smaller grid/place sizes and fewer frequencies
    params.n_grids_all = [12, 9, 6]
    params.grid2phase = 3
    params.n_phases_all = [int(n_grid / params.grid2phase) for n_grid in params.n_grids_all]
    params.tot_phases = sum(params.n_phases_all)
    params.n_freq = len(params.n_phases_all)
    params.g_size = sum(params.n_grids_all)
    params.n_place_all = [p * params.s_size_comp for p in params.n_phases_all]
    params.p_size = sum(params.n_place_all)
    params.s_size_comp_hidden = 10 * params.s_size_comp
    params.prediction_freq = 0
    params.freqs = sorted([0.01, 0.7, 0.91, 0.97, 0.99, 0.9995])[:params.n_freq]

    # Smaller MLP for transition and input
    params.d_mixed_size = 10

    # Fewer envs saved for summaries
    params.n_envs_save = min(params.n_envs_save, 2)

    # Rebuild masks/connectivity for new shapes
    params.R_f_F = parameters.connectivity_matrix(parameters.conn_hierarchical, params.freqs)
    params.R_f_F_inv = parameters.connectivity_matrix(parameters.conn_all2all, params.freqs)
    params.mask_p = parameters.get_mask(params.n_place_all, params.n_place_all,
                                        parameters.transpose_connectivity(params.R_f_F))
    params.R_G_F_f = parameters.connectivity_matrix(parameters.conn_hierarchical, params.freqs)
    params.mask_g = parameters.get_mask(params.n_grids_all, params.n_grids_all, params.R_G_F_f)
    params.n_recurs = params.n_freq
    params.max_attractor_its = [params.n_recurs - f for f in range(params.n_freq)]
    params.max_attractor_its_inv = [params.n_recurs for _ in range(params.n_freq)]
    params.attractor_freq_iterations = [[f for f in range(params.n_freq) if r < params.max_attractor_its[f]]
                                        for r in range(params.n_recurs)]
    params.attractor_freq_iterations_inv = [[f for f in range(params.n_freq) if r < params.max_attractor_its_inv[f]]
                                            for r in range(params.n_recurs)]
    params.R_f_F_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F]
    params.R_f_F_inv_ = [list(np.where(x[:params.n_freq])[0]) for x in params.R_f_F_inv]

    # Adjust training steps/summaries
    params.train_iters = min(params.train_iters, 5000)
    params.sum_int = max(50, params.sum_int)
    params.sum_int_inferences = max(200, params.sum_int_inferences)

    # Re-derive environment-specific params for possibly changed world
    params = parameters.get_env_params(params, width=None, height=None)
# Save parameters
np.save(os.path.join(save_path, 'params'), dict(params))

tf.keras.backend.set_floatx('float32')
params.precision = model_utils.precision

# Create instance of TEM with those parameters
model = tem.TEM(params)

# Create a logger to write log output to file
logger_sums = data_utils.make_logger(run_path, 'summaries')
logger_envs = data_utils.make_logger(run_path, 'env_details')
# Create a tensor board to stay updated on training progress. Start tensorboard with tensorboard --logdir=runs
summary_writer = tf.summary.create_file_writer(train_path)

# Make an ADAM optimizer for TEM
optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate_max)

# Specify training setup
debug_data = False
optimise = True
debug = False
profile = False
tf.config.run_functions_eagerly(debug or debug_data or not params.graph_mode)
if profile or debug:
    params.train_iters = 11
    params.save_walk = 1


@tf.function  # (experimental_compile=True)
def train_step(model_, model_inputs_):
    # everything entering this function should already be a tensor!
    with tf.GradientTape() as tape:
        print('start_model')
        variables_, re_input_ = model_(model_inputs_, training=True)
        # collate inputs for model
        losses_ = tem.compute_losses(model_inputs_, variables_, model_.trainable_variables, params)
        print('compute_losses_done')
    if optimise:
        gradients_ = tape.gradient(losses_.train_loss, model_.trainable_variables)
        capped_grads = [(tf.clip_by_norm(grad, 2), var) if grad is not None else (grad, var) for grad, var in
                        zip(gradients_, model_.trainable_variables)]
        optimizer.apply_gradients(capped_grads)
    print('train_step_optimiser_done')
    return variables_, re_input_, losses_


@tf.function  # (experimental_compile=True)
def test_step(model_, model_inputs_):
    variables_, re_input_ = model_(model_inputs_, training=False)
    return variables_, re_input_


# initialise dictionary to contain environments and data info
train_dict = data_utils.get_initial_data_dict(params) # keys: two_hot_table, env_steps, curric_env, hebb, variables, walk_data, bptt_data
# two_hot_table: a list of tuples. Each tuple is a array with all zeros except two ones. I guess this is the sensory stimulus.

msg = 'Training Started'
logger_sums.info(msg)
logger_envs.info(msg)

# Add total training time tracking
total_start_time = time.time()

for train_i in range(params.train_iters):
    # INITIALISE ENVIRONMENT AND INPUT VARIABLES
    if sum(train_dict.env_steps == 0) > 0:
        msg = str(sum(train_dict.env_steps == 0)) + ' New Environments ' + str(train_i) + ' ' + str(
            train_i * params.seq_len)
        logger_envs.info(msg)

    # Get scaling parameters
    scalings = parameters.get_scaling_parameters(train_i, params)
    optimizer.lr = scalings.l_r

    data_start_time = time.time()
    # collect batch-specific environment data
    train_dict = data_utils.data_step(train_dict, params)
    # convert all inputs to tensors - otherwise re-build graph every time
    inputs_tf = model_utils.inputs_2_tf(train_dict.inputs, train_dict.hebb, scalings, params.n_freq)

    model_start_time = time.time()
    if debug_data:
        continue
    elif profile and train_i > 0:
        # start profiler after graph set-up
        if train_i == 1:
            tf.profiler.experimental.start(train_path)
        # profile
        with tf.profiler.experimental.Trace('train', step_num=train_i, _r=1):
            variables, re_input, losses = train_step(model, inputs_tf)
    else:
        variables, re_input, losses = train_step(model, inputs_tf)
    stop_time = time.time()

    # Update logging info
    train_dict.curric_env.data_step_time = model_start_time - data_start_time
    train_dict.curric_env.model_step_time = stop_time - model_start_time
    msg = 'Step {:.0f}, data time : {:.4f} , model time {:.4f}'.format(train_i, model_start_time - data_start_time,
                                                                       stop_time - model_start_time)
    logger_envs.info(msg)

    # feeding in correct initial states
    re_input = model_utils.tf2numpy(re_input)
    train_dict.variables.gs, train_dict.variables.x_s, train_dict.hebb.a_rnn, train_dict.hebb.a_rnn_inv = \
        re_input.g, re_input.x_s, re_input.a_rnn, re_input.a_rnn_inv

    # Log training progress summary statistics
    if train_i % params.sum_int == 0 and train_i > 0:
        accuracies = model_utils.compute_accuracies(inputs_tf.x, variables.pred, params)
        summaries = model_utils.make_summaries(losses, accuracies, scalings, variables, train_dict.curric_env,
                                               train_dict.inputs.seq_index, params)

        for key_, val_ in summaries.items():
            with summary_writer.as_default():
                tf.summary.scalar(key_, val_, step=train_i)
        summary_writer.flush()

        msg = "MaxHebb={:.5f}, T={:.2f}, F={:.2f}, L={:.2f}, train_i={:.0f}, total_steps={:.0f}".format(
            np.max(np.max(np.abs(train_dict.hebb.a_rnn))), scalings.temp, scalings.forget, scalings.h_l,
            train_i, train_i * params.seq_len)
        logger_sums.info(msg)
        msg = ("it={:.0f}, lxP={:.2f}, lxG={:.2f}, lxGt={:.2f}, lp={:.2f}, lg={:.2f}, aP={:.2f}, aGinf={:.2f}, "
               + "aGt={:.2f}").format(train_i, losses.lx_p, losses.lx_g, losses.lx_gt, losses.lp,
                                      losses.lg, accuracies.p, accuracies.g, accuracies.gt)
        logger_sums.info(msg)

    # Log current model performance summaries - requires running full environments from scratch
    if train_i % params.sum_int_inferences == 0 and train_i > 0:
        start_time = time.time()
        data_utils.summary_inferences(train_i, model, test_step, summary_writer, params)
        logger_envs.info(
            "summary inference time {:.2f}, train_i={:.2f}, total_steps={:.2f}".format(time.time() - start_time,
                                                                                       train_i,
                                                                                       train_i * params.seq_len))

    # Save model parameters which can be loaded later to analyse model 
    if train_i % params.save_interval == 0:  # and train_i > 0:
        start_time = time.time()
        # data_utils.save_model_outputs(test_step, train_i, save_path, params)

        # save model checkpoint
        model.save_weights(model_path + '/tem_' + str(train_i))
        logger_sums.info("save data time {:.2f}, train_i={:.2f}, total_steps={:.2f}".format(time.time() - start_time,
                                                                                            train_i,
                                                                                            train_i * params.seq_len))

    # save model
    # if train_i % params.save_model == 0:
    #    model.save(model_path + '/tem_' + str(train_i))
if profile:
    tf.profiler.experimental.stop()

# Calculate and print total training time
total_training_time = time.time() - total_start_time
print(f'Finished training in {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)')


def check_inputs_modified(x, y):
    if isinstance(x, tf.Tensor):
        print(tf.reduce_sum((x - y) ** 2))
    elif isinstance(x, dict) or isinstance(x, model_utils.DotDict):
        for key, value in x.items():
            check_inputs_modified(value, y[key])
    elif isinstance(x, list):
        for i, a in enumerate(x):
            check_inputs_modified(a, y[i])
    else:
        print(type(x), type(y))
    return
