#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal, self-contained demo for the Rectangle environment.

Usage examples:

  - Basic walk (no plotting):
      python tem_tf2/env_test.py --width 5 --height 4 --steps 40

  - With plotting (requires matplotlib):
      python tem_tf2/env_test.py --width 8 --height 6 --steps 80 --plot

  - Toroidal world (wrap on edges):
      python tem_tf2/env_test.py --width 6 --height 6 --torus

This script avoids TensorFlow and the full TEM model. It only uses numpy
and the environment logic in environments.py to help you learn the API.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Tuple

import numpy as np

from parameters import default_params, combins_table, onehot2twohot, get_scaling_parameters
from environments import Rectangle, sample_data
from tem_model import TEM
import model_utils


def build_rectangle_env(width: int, height: int, seed: Optional[int], torus: bool):
    """
    Build and initialise a Rectangle environment instance.
    """
    if seed is not None:
        np.random.seed(seed)

    # Create parameter object scoped to a simple rectangle world with batch_size=1
    params = default_params(width=width, height=height, world_type='rectangle', batch_size=1)

    # Construct the environment and initialise its graph/world
    env = Rectangle(params, width=width, height=height)
    env.world(torus=torus)
    env.state_data()  # assigns a random sensory label to each state
    return env


def simulate_walk(env: Rectangle, steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run a walk for a given number of steps. Returns (positions, one_hot_actions_over_time).
    """
    env.walk_len = steps
    positions, actions_one_hot = env.walk() # action is determined by the environment.
    return positions, actions_one_hot


def run_model_forward(env: Rectangle, positions: np.ndarray, actions_one_hot: np.ndarray):
    """
    Build minimal inputs and run a single forward pass of TEM (no training).
    Prints shapes and example probabilities.
    """
    par = env.par

    # Align model sequence length with our walk length
    steps = int(positions.shape[0])
    par.seq_len = steps

    # Batch size = 1
    batch_size = 1

    # Observations (one-hot) from positions via env.states_mat
    xs_1 = sample_data(positions, env.states_mat, par.s_size)  # (s_size, steps)
    xs = np.expand_dims(xs_1, axis=0).astype(np.float32)  # (1, s_size, steps)

    # Two-hot embedding of observations
    two_hot_tbl = combins_table(par.s_size_comp, 2)
    xs_two_hot = onehot2twohot(xs, two_hot_tbl, par.s_size_comp).astype(np.float32)  # (1, s_size_comp, steps)

    # Initial filtered sensory state per frequency (flattened across freqs)
    x_s_init = np.zeros((batch_size, par.s_size_comp * par.n_freq), dtype=np.float32)

    # Initial grid state
    gs_init = np.zeros((batch_size, par.g_size), dtype=np.float32)

    # Directions one-hot: (1, n_actions, steps)
    ds = np.expand_dims(actions_one_hot, axis=0).astype(np.float32)

    # Misc inputs required by inputs_2_tf
    seq_index = np.zeros(batch_size, dtype=np.float32)
    s_visited = np.ones((batch_size, steps), dtype=np.float32)
    positions_b = np.expand_dims(positions, axis=0).astype(np.float32)
    reward_val = np.array([getattr(env, 'reward_value', 0.0)], dtype=np.float32)
    reward_pos = np.zeros((batch_size, 1), dtype=np.float32)

    # Hebbian matrices
    hebb = model_utils.DotDict({
        'a_rnn': np.zeros((batch_size, par.p_size, par.p_size), dtype=np.float32),
        'a_rnn_inv': np.zeros((batch_size, par.p_size, par.p_size), dtype=np.float32),
    })

    # Pack inputs and convert to TF tensors
    inputs_np = model_utils.DotDict({
        'xs': xs, # (1, s_size, steps) one-hot observations. From positions via env.states_mat. This should be images. You need to provide this.
        'x_s': x_s_init, # (1, s_size, steps) initial filtered sensory state per frequency (flattened across freqs). This should be filtered over x_s. You need to provide this.
        'xs_two_hot': xs_two_hot, # (1, s_size_comp, steps) two-hot embedding of observations. This should be some embedding of the images. You need to provide this.
        'gs': gs_init, # (1, g_size) initial grid state. Just make it all zeros.
        'ds': ds, # (1, n_actions, steps) one-hot actions. You need to provide this, but I think make it all zeros are fine for you since the working memory task has no real input actions, only observations.
        'seq_index': seq_index, # (1,) per-batch environment "episode" index within the curriculum. Zero for a fresh run.
        's_visited': s_visited, # (1, steps) mask of whether each timestep's position wwas previously visited (1/0). Used to mask losses.
        'positions': positions_b, # this will not be used by the model. You need to provide this. But your working memory task is simple, just make it (1, 2, ...) would be fine.
        'reward_val': reward_val, # ignore this if you are not using reward training
        'reward_pos': reward_pos, # ignore this if you are not using reward training
    }) # Together, the parameters you need to provide are xs, x_s, xs_two_hot, ds, positions.
    scalings = get_scaling_parameters(0, par)
    inputs_tf = model_utils.inputs_2_tf(inputs_np, hebb, scalings, par.n_freq)

    # Build and run model
    model = TEM(par)
    variables, _ = model(inputs_tf, training=False)
    # g.g[i]: inferred/posterior grids (data + transition + optional memory)
    # g.g_gen[i]: predicted grids (no data correction)
    # p.p[i]: inferred places from sensory input and grids (with data correction)
    # p.p_g[i]: place cell generated from g.g[i]
    # p.p_x[i]: place cell generated from x_s[i]
    # x_s[i]: temporally filtered sensory state, used as model's rolling sensory memory for the next step.
    # pred.x_p[i]: from inferred places p
    # pred.x_g[i]: from generated places p_g
    # pred.x_gt[i]: from generated places using transition-only grids (no data)
    # logits.x_p[i], logits.x_g[i], logits.x_gt[i]: pre-softmax logits.

    # Convert a few outputs to numpy and print quick summary
    variables_np = model_utils.tf2numpy(variables)
    preds = variables_np.pred

    def topk(prob_vec, k=3):
        idxs = np.argsort(prob_vec)[::-1][:k]
        return list(zip(idxs.tolist(), prob_vec[idxs].round(4).tolist()))

    print("\n=== Model forward (no training) ===")
    print(f"x_p shape per step: {preds.x_p[0].shape}; x_g: {preds.x_g[0].shape}; x_gt: {preds.x_gt[0].shape}")

    # Per-timestep comparisons (sampled timesteps)
    for t in [0, min(steps - 1, 1), min(steps - 1, 5), steps - 1]:
        p_p = preds.x_p[t][0]
        p_g = preds.x_g[t][0]
        p_gt = preds.x_gt[t][0]
        gt_idx = int(np.argmax(xs[0, :, t]))
        top1_p_idx, top1_g_idx, top1_gt_idx = int(np.argmax(p_p)), int(np.argmax(p_g)), int(np.argmax(p_gt))
        top1_p_prob, top1_g_prob, top1_gt_prob = float(np.max(p_p)), float(np.max(p_g)), float(np.max(p_gt))
        print(
            f"t={t:02d}  gt={gt_idx}  "
            f"x_p: top1={top1_p_idx} (p={top1_p_prob:.3f})  "
            f"x_g: top1={top1_g_idx} (p={top1_g_prob:.3f})  "
            f"x_gt: top1={top1_gt_idx} (p={top1_gt_prob:.3f})"
        )
        print(
            f"           match?  x_p={top1_p_idx==gt_idx}  x_g={top1_g_idx==gt_idx}  x_gt={top1_gt_idx==gt_idx}"
        )

    # Sequence accuracy over all steps
    gt_all = np.argmax(xs[0], axis=0)
    pred_p_all = np.array([int(np.argmax(preds.x_p[t][0])) for t in range(steps)])
    pred_g_all = np.array([int(np.argmax(preds.x_g[t][0])) for t in range(steps)])
    pred_gt_all = np.array([int(np.argmax(preds.x_gt[t][0])) for t in range(steps)])

    acc_p = float(np.mean(pred_p_all == gt_all))
    acc_g = float(np.mean(pred_g_all == gt_all))
    acc_gt = float(np.mean(pred_gt_all == gt_all))
    print(f"Accuracy over {steps} steps  x_p={acc_p:.3f}  x_g={acc_g:.3f}  x_gt={acc_gt:.3f}")

def maybe_plot(env: Rectangle, positions: np.ndarray, save_path: Optional[str]):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("[plot] matplotlib not available (", exc, ") — skipping plot.")
        return

    xs, ys = env.get_node_positions()
    path_x = xs[positions]
    path_y = ys[positions]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(xs, ys, s=20, c='#888888', alpha=0.7, label='states')
    ax.plot(path_x, path_y, '-o', ms=3, lw=1.0, c='#1f77b4', label='trajectory')
    ax.scatter([path_x[0]], [path_y[0]], c='green', s=60, zorder=3, label='start')
    ax.scatter([path_x[-1]], [path_y[-1]], c='red', s=60, zorder=3, label='end')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Rectangle {env.width}x{env.height} walk ({len(positions)} steps)")
    ax.legend(loc='best')
    ax.grid(True, ls=':', alpha=0.4)


    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[plot] Saved to {save_path}")
    else:
        plt.show()


def print_quick_info(env: Rectangle, positions: np.ndarray, actions_one_hot: np.ndarray, max_print: int = 8):
    print("\n=== Rectangle environment info ===")
    print(f"Size: width={env.width}, height={env.height}, n_states={env.width * env.height}")
    print(f"Actions (n={env.n_actions}): {env.rels}")
    print(f"Start state: {int(positions[0])}")
    print(f"Unique states visited: {len(np.unique(positions))}")
    print("Adjacency row sums (first 5 states):",
          [int(np.sum(env.adj[i])) for i in range(min(5, env.width * env.height))])
    print("Example transitions and relations:")
    for t in range(min(max_print, len(positions) - 1)):
        s1 = int(positions[t])
        s2 = int(positions[t + 1])
        rel_idx, rel_type = env.relation(s1, s2)
        print(f"  t={t:02d}: {s1} -> {s2}  relation={rel_type} (idx {rel_idx})")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn and test the Rectangle environment")
    parser.add_argument('--width', type=int, default=10, help='Grid width')
    parser.add_argument('--height', type=int, default=10, help='Grid height')
    parser.add_argument('--steps', type=int, default=60, help='Number of walk steps')
    parser.add_argument('--torus', action='store_true', help='Wrap around edges (toroidal world)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Show a plot of the trajectory (matplotlib)')
    parser.add_argument('--save-plot', type=str, default=None, help='If set, save the plot to this path')
    parser.add_argument('--run-model', action='store_true', help='Run a single forward pass of TEM (no training)')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    env = build_rectangle_env(width=args.width, height=args.height, seed=args.seed, torus=args.torus)
    positions, actions_one_hot = simulate_walk(env, steps=args.steps)
    xs = sample_data(positions, env.states_mat, env.par.s_size) # one-hot encoding of sensory at each time-step

    print_quick_info(env, positions, actions_one_hot)

    if args.run_model:
        run_model_forward(env, positions, actions_one_hot)

    if args.plot or args.save_plot:
        maybe_plot(env, positions, save_path=args.save_plot)

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))


