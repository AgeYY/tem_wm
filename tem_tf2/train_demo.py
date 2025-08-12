#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal training demo for TEM on a tiny Rectangle environment (TF2).

This script mirrors the simplicity of `env_test.py`, but adds a very small
training loop to show how to optimise TEM end-to-end without the full
training pipeline.

Example:
  python tem_tf2/train_demo.py --width 6 --height 5 --steps 40 --iters 200 --fast --seed 0

Notes:
- Uses a single environment sequence per batch (no dataset/loader).
- Keeps the model small in --fast mode so it runs quickly on CPU.
- Updates the re-input state (x_s, g, Hebbian) between steps like the full trainer.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

import parameters
from parameters import (
    default_params,
    combins_table,
    onehot2twohot,
    get_scaling_parameters,
)
from environments import Rectangle, sample_data
from tem_model import TEM, compute_losses
import model_utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def build_fast_params(width: int, height: int, batch_size: int, steps: int, fast: bool) -> object:
    par = default_params(width=width, height=height, world_type='rectangle', batch_size=batch_size)
    par.seq_len = int(steps)
    par.use_reward = False
    par.tf_range = True
    par.graph_mode = True
    par.n_envs = batch_size

    if fast:
        par.s_size_comp = 6
        s_size_fast = len(combins_table(par.s_size_comp, 2))
        par.s_size = s_size_fast
        par.two_hot_mat = onehot2twohot(
            np.expand_dims(np.eye(par.s_size), axis=0),
            combins_table(par.s_size_comp, 2),
            par.s_size_comp,
        )

        par.n_grids_all = [12, 9, 6]
        par.grid2phase = 3
        par.n_phases_all = [int(n_grid / par.grid2phase) for n_grid in par.n_grids_all]
        par.tot_phases = sum(par.n_phases_all)
        par.n_freq = len(par.n_phases_all)
        par.g_size = sum(par.n_grids_all)
        par.n_place_all = [p * par.s_size_comp for p in par.n_phases_all]
        par.p_size = sum(par.n_place_all)
        par.s_size_comp_hidden = 10 * par.s_size_comp
        par.prediction_freq = 0
        par.freqs = sorted([0.01, 0.7, 0.91, 0.97, 0.99, 0.9995])[:par.n_freq]
        par.d_mixed_size = 10
        par.n_envs_save = min(par.n_envs_save, 2)

        # Rebuild masks/connectivity for new shapes
        par.R_f_F = parameters.connectivity_matrix(parameters.conn_hierarchical, par.freqs)
        par.R_f_F_inv = parameters.connectivity_matrix(parameters.conn_all2all, par.freqs)
        par.mask_p = parameters.get_mask(
            par.n_place_all,
            par.n_place_all,
            parameters.transpose_connectivity(par.R_f_F),
        )
        par.R_G_F_f = parameters.connectivity_matrix(parameters.conn_hierarchical, par.freqs)
        par.mask_g = parameters.get_mask(par.n_grids_all, par.n_grids_all, par.R_G_F_f)
        par.n_recurs = par.n_freq
        par.max_attractor_its = [par.n_recurs - f for f in range(par.n_freq)]
        par.max_attractor_its_inv = [par.n_recurs for _ in range(par.n_freq)]
        par.attractor_freq_iterations = [
            [f for f in range(par.n_freq) if r < par.max_attractor_its[f]] for r in range(par.n_recurs)
        ]
        par.attractor_freq_iterations_inv = [
            [f for f in range(par.n_freq) if r < par.max_attractor_its_inv[f]] for r in range(par.n_recurs)
        ]
        par.R_f_F_ = [list(np.where(x[:par.n_freq])[0]) for x in par.R_f_F]
        par.R_f_F_inv_ = [list(np.where(x[:par.n_freq])[0]) for x in par.R_f_F_inv]

    return par


def build_batch_rectangle_envs(
    batch_size: int, width: int, height: int, steps: int, seed: Optional[int], torus: bool, par
) -> Tuple[np.ndarray, np.ndarray, list[Rectangle]]:
    if seed is not None:
        np.random.seed(seed)

    envs = []
    positions_b = np.zeros((batch_size, steps), dtype=np.int32)
    actions_b = np.zeros((batch_size, par.n_actions, steps), dtype=np.float32)

    for b in range(batch_size):
        env = Rectangle(par, width=width, height=height)
        env.world(torus=torus)
        env.state_data()
        env.walk_len = steps
        pos, act_oh = env.walk()
        envs.append(env)
        positions_b[b] = pos.astype(np.int32)
        actions_b[b] = act_oh.astype(np.float32)

    return positions_b, actions_b, envs


def make_inputs_from_walk(
    par, envs: list[Rectangle], positions_b: np.ndarray, actions_b: np.ndarray
) -> Tuple[model_utils.DotDict, model_utils.DotDict]:
    batch_size, steps = positions_b.shape
    xs = np.zeros((batch_size, par.s_size, steps), dtype=np.float32)
    for b, env in enumerate(envs):
        xs[b] = sample_data(positions_b[b], env.states_mat, par.s_size).astype(np.float32)

    xs_two_hot = onehot2twohot(xs, combins_table(par.s_size_comp, 2), par.s_size_comp).astype(np.float32)

    x_s_init = np.zeros((batch_size, par.s_size_comp * par.n_freq), dtype=np.float32)
    gs_init = np.zeros((batch_size, par.g_size), dtype=np.float32)

    seq_index = np.zeros(batch_size, dtype=np.float32)
    s_visited = np.ones((batch_size, steps), dtype=np.float32)
    reward_val = np.zeros((batch_size,), dtype=np.float32)
    reward_pos = np.zeros((batch_size, 1), dtype=np.float32)

    inputs_np = model_utils.DotDict(
        {
            'xs': xs,
            'x_s': x_s_init,
            'xs_two_hot': xs_two_hot,
            'gs': gs_init,
            'ds': actions_b,
            'seq_index': seq_index,
            's_visited': s_visited,
            'positions': positions_b.astype(np.float32),
            'reward_val': reward_val,
            'reward_pos': reward_pos,
        }
    )

    hebb = model_utils.DotDict(
        {
            'a_rnn': np.zeros((batch_size, par.p_size, par.p_size), dtype=np.float32),
            'a_rnn_inv': np.zeros((batch_size, par.p_size, par.p_size), dtype=np.float32),
        }
    )

    return inputs_np, hebb


def train_once(model: TEM, inputs_tf, par, optimizer: tf.keras.optimizers.Optimizer):
    with tf.GradientTape() as tape:
        variables, re_input = model(inputs_tf, training=True)
        losses = compute_losses(inputs_tf, variables, model.trainable_variables, par)
    grads = tape.gradient(losses.train_loss, model.trainable_variables)
    capped_grads = [tf.clip_by_norm(g, 2.0) if g is not None else None for g in grads]
    optimizer.apply_gradients([(g, v) for g, v in zip(capped_grads, model.trainable_variables) if g is not None])
    return variables, re_input, losses


def accuracy_from_preds(xs_t, preds, par) -> Tuple[float, float, float]:
    accs = model_utils.compute_accuracies(xs_t, preds, par)
    return float(accs['p']), float(accs['g']), float(accs['gt'])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Minimal TEM training demo')
    p.add_argument('--width', type=int, default=4)
    p.add_argument('--height', type=int, default=4)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--fast', action='store_true')
    p.add_argument('--torus', action='store_true')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--save', type=str, default=None, help='Optional path prefix to save model weights')
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    # Build params (fast keeps things tiny for quick demo runs)
    par = build_fast_params(args.width, args.height, args.batch_size, args.steps, args.fast)
    tf.keras.backend.set_floatx('float32')
    par.precision = model_utils.precision

    # Build a small batch of environments and a single walk per env
    positions_b, actions_b, envs = build_batch_rectangle_envs(
        args.batch_size, args.width, args.height, args.steps, args.seed, args.torus, par
    )

    # Construct initial inputs and Hebbian state
    inputs_np, hebb = make_inputs_from_walk(par, envs, positions_b, actions_b)

    # Build model and optimiser
    model = TEM(par)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # Training loop
    printed_param_count = False
    for it in range(args.iters):
        scalings = get_scaling_parameters(it, par)
        inputs_tf = model_utils.inputs_2_tf(inputs_np, hebb, scalings, par.n_freq)

        variables, re_input, losses = train_once(model, inputs_tf, par, optimizer)

        if not printed_param_count:
            try:
                param_count = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
                print(f"Trainable parameters (TEM): {param_count:,}")
            except Exception:
                pass
            printed_param_count = True

        # Update re-input state (keeps filtered sensory, grids, and Hebbian)
        re_np = model_utils.tf2numpy(re_input)
        inputs_np.x_s = re_np.x_s
        inputs_np.gs = re_np.g
        hebb.a_rnn = re_np.a_rnn
        hebb.a_rnn_inv = re_np.a_rnn_inv

        # if (it + 1) % max(1, args.iters // 10) == 0 or it == 0:
        if True:
            xs_t = model_utils.inputs_2_tf(inputs_np, hebb, scalings, par.n_freq).x
            acc_p, acc_g, acc_gt = accuracy_from_preds(xs_t, variables.pred, par)
            print(
                f"iter={it+1:04d} loss={float(losses.train_loss):.4f} "
                f"lx_p={float(getattr(losses, 'lx_p', 0.0)):.4f} "
                f"lx_g={float(getattr(losses, 'lx_g', 0.0)):.4f} "
                f"lx_gt={float(getattr(losses, 'lx_gt', 0.0)):.4f} "
                f"acc_p={acc_p:.2f} acc_g={acc_g:.2f} acc_gt={acc_gt:.2f}"
            )

    # Optional save
    if args.save:
        model.save_weights(args.save)
        print(f"Saved weights to {args.save}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


