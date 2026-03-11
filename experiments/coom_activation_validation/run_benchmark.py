#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp")

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack, ResizeObservation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from COOM.env.builder import build_multi_discrete_actions, make_env
from COOM.utils.config import Scenario, default_wrapper_config


sns.set_theme(style="whitegrid", context="talk")


@dataclass(frozen=True)
class Benchmark:
    scenario: Scenario
    task: str

    @property
    def name(self) -> str:
        return f"{self.scenario.name.lower()}-{self.task}"


def parse_benchmark(text: str) -> Benchmark:
    try:
        scenario_name, task = text.split("-", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid benchmark '{text}'. Expected format like 'pitfall-default'."
        ) from exc
    scenario = Scenario[scenario_name.upper()]
    return Benchmark(scenario=scenario, task=task)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)


class CastObsToFloat32(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32)


class RescaleToMinusOneOne(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        return obs / 255.0 * 2.0 - 1.0


class CollapseFrameStack(ObservationWrapper):
    # COOM's RGBStack wrapper is only applied behind the `lstm` flag.
    # This wrapper converts [stack, H, W, C] into [H, W, stack*C] for CNNs.
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        if len(obs_shape) != 4:
            raise ValueError(f"Expected stacked observation with 4 dims, got {obs_shape}.")
        stack, height, width, channels = obs_shape
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(height, width, stack * channels),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        return np.transpose(obs, (1, 2, 0, 3)).reshape(obs.shape[1], obs.shape[2], -1)


class ChannelsFirst(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        height, width, channels = env.observation_space.shape
        self.observation_space = Box(
            low=-1.0,
            high=1.0,
            shape=(channels, height, width),
            dtype=np.float32,
        )

    def observation(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        return np.transpose(obs, (2, 0, 1))


class DualTimescaleActivation(nn.Module):
    # Operationalizes the paper idea with local novelty, a utility trace,
    # and a slow anchor blended into a GELU/SELU-style mixture.
    def __init__(self) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(th.tensor(-1.5))
        self.log_scale = nn.Parameter(th.tensor(-2.0))
        self.register_buffer("running_mean", th.zeros(1))
        self.register_buffer("running_var", th.ones(1))
        self.register_buffer("utility_trace", th.zeros(1))
        self.register_buffer("slow_alpha", th.zeros(1))
        self.register_buffer("slow_scale", th.zeros(1))
        self.register_buffer("initialized", th.zeros(1))
        self.stat_momentum = 0.05
        self.utility_momentum = 0.02
        self.anchor_momentum = 0.002
        self.beta_novelty = 1.75
        self.beta_utility = 1.0
        self.eps = 1e-5

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.training:
            with th.no_grad():
                x_detached = x.detach()
                batch_mean = x_detached.mean()
                batch_var = x_detached.var(unbiased=False).clamp_min(self.eps)
                novelty = (batch_mean - self.running_mean).abs() / th.sqrt(self.running_var + self.eps)
                utility = x_detached.abs().mean()

                if self.initialized.item() == 0:
                    self.slow_alpha.copy_(self.log_alpha.detach())
                    self.slow_scale.copy_(self.log_scale.detach())
                    self.initialized.fill_(1.0)

                self.running_mean.lerp_(batch_mean, self.stat_momentum)
                self.running_var.lerp_(batch_var, self.stat_momentum)
                self.utility_trace.lerp_(utility, self.utility_momentum)
                self.slow_alpha.lerp_(self.log_alpha.detach(), self.anchor_momentum)
                self.slow_scale.lerp_(self.log_scale.detach(), self.anchor_momentum)
        else:
            novelty = th.zeros(1, device=x.device)

        utility_gate = th.sigmoid(1.75 * self.utility_trace)
        alpha_raw = (1.0 - utility_gate) * self.log_alpha + utility_gate * self.slow_alpha
        scale_raw = (1.0 - utility_gate) * self.log_scale + utility_gate * self.slow_scale
        alpha = 1.0 + 0.2 * th.tanh(alpha_raw)
        scale = 0.05 + 0.15 * th.sigmoid(scale_raw)

        mix_gate = th.sigmoid(-2.5 + self.beta_novelty * novelty - self.beta_utility * self.utility_trace)
        gelu_branch = F.gelu(x)
        selu_like_branch = F.selu(alpha * x)
        return gelu_branch + scale * mix_gate * (selu_like_branch - gelu_branch)


def build_activation(kind: str) -> nn.Module:
    if kind == "gelu":
        return nn.GELU()
    if kind == "proposed":
        return DualTimescaleActivation()
    raise KeyError(f"Unsupported activation kind: {kind}")


class COOMCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, activation_kind: str = "gelu", features_dim: int = 256) -> None:
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.GELU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            build_activation(activation_kind),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def make_base_env(benchmark: Benchmark, seed: int):
    wrap_conf = dict(default_wrapper_config)
    wrap_conf.update(
        {
            "augment": False,
            "resize": False,
            "rescale": False,
            "normalize_observation": False,
            "frame_stack": False,
            "lstm": False,
            "record": False,
        }
    )
    doom_kwargs = {
        "env": benchmark.task,
        "task_idx": 0,
        "action_space_fn": build_multi_discrete_actions,
        "render": False,
        "test_only": False,
        "seed": seed,
    }
    env = make_env(
        benchmark.scenario,
        task=benchmark.task,
        doom_kwargs=doom_kwargs,
        wrapper_config=wrap_conf,
    )
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)
    env = CollapseFrameStack(env)
    env = CastObsToFloat32(env)
    env = RescaleToMinusOneOne(env)
    env = ChannelsFirst(env)
    return env


def make_vec_env(benchmark: Benchmark, seed: int):
    return DummyVecEnv([lambda: make_base_env(benchmark, seed)])


def raw_obs_to_model_obs(obs: np.ndarray) -> np.ndarray:
    return obs


def env_success(env) -> float:
    current = env
    while hasattr(current, "env"):
        current = current.env
    return float(current.get_success())


def unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def evaluate_model(
    model: PPO,
    benchmark: Benchmark,
    seed: int,
    n_episodes: int,
) -> Dict[str, float]:
    rewards: List[float] = []
    lengths: List[int] = []
    successes: List[float] = []
    env = make_base_env(benchmark, seed + 10_000)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_len = 0
        while not done and not truncated:
            action, _ = model.predict(raw_obs_to_model_obs(obs), deterministic=True)
            obs, reward, done, truncated, _ = env.step(int(action))
            ep_reward += float(reward)
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(env_success(env))
    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_success": float(np.mean(successes)),
        "mean_ep_length": float(np.mean(lengths)),
    }


class EvaluationLogger(BaseCallback):
    def __init__(
        self,
        benchmark: Benchmark,
        seed: int,
        eval_every: int,
        eval_episodes: int,
        csv_path: Path,
    ) -> None:
        super().__init__()
        self.benchmark = benchmark
        self.seed = seed
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.csv_path = csv_path
        self.rows: List[Dict[str, float | int]] = []
        self.best_reward = -math.inf
        self.best_success = -math.inf
        self.best_reward_path = self.csv_path.parent / "best_reward_model"
        self.best_success_path = self.csv_path.parent / "best_success_model"

    def _write_rows(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            return
        fieldnames = list(self.rows[0].keys())
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def _evaluate(self) -> None:
        metrics = evaluate_model(
            model=self.model,
            benchmark=self.benchmark,
            seed=self.seed,
            n_episodes=self.eval_episodes,
        )
        row = {
            "timesteps": int(self.num_timesteps),
            "mean_reward": metrics["mean_reward"],
            "std_reward": metrics["std_reward"],
            "mean_success": metrics["mean_success"],
            "mean_ep_length": metrics["mean_ep_length"],
        }
        self.rows.append(row)
        if row["mean_reward"] > self.best_reward:
            self.best_reward = float(row["mean_reward"])
            self.model.save(self.best_reward_path)
        if row["mean_success"] > self.best_success:
            self.best_success = float(row["mean_success"])
            self.model.save(self.best_success_path)
        self._write_rows()

    def _on_training_start(self) -> None:
        self._evaluate()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._evaluate()
        return True

    def _on_training_end(self) -> None:
        if not self.rows or self.rows[-1]["timesteps"] != int(self.num_timesteps):
            self._evaluate()


def build_model(activation_kind: str, env, seed: int) -> PPO:
    activation_fn = nn.GELU if activation_kind == "gelu" else DualTimescaleActivation
    policy_kwargs = {
        "features_extractor_class": COOMCNNExtractor,
        "features_extractor_kwargs": {"activation_kind": activation_kind, "features_dim": 256},
        "activation_fn": activation_fn,
        "net_arch": [256, 256],
        "normalize_images": False,
    }
    return PPO(
        "CnnPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="cpu",
    )


def train_run(
    benchmark: Benchmark,
    activation_kind: str,
    seed: int,
    timesteps: int,
    eval_every: int,
    eval_episodes: int,
    output_root: Path,
) -> Path:
    set_global_seed(seed)
    run_dir = output_root / "runs" / benchmark.name / activation_kind / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(benchmark, seed)
    model = build_model(activation_kind, env, seed)
    callback = EvaluationLogger(
        benchmark=benchmark,
        seed=seed,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        csv_path=run_dir / "eval_metrics.csv",
    )
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model.save(run_dir / "model")
    env.close()

    summary = {
        "benchmark": benchmark.name,
        "activation": activation_kind,
        "seed": seed,
        "timesteps": timesteps,
        "final_eval": callback.rows[-1],
        "best_success": max(row["mean_success"] for row in callback.rows),
        "best_reward": max(row["mean_reward"] for row in callback.rows),
        "best_reward_model": str(callback.best_reward_path.with_suffix(".zip")),
        "best_success_model": str(callback.best_success_path.with_suffix(".zip")),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir


def aggregate_results(output_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for csv_path in output_root.glob("runs/*/*/seed_*/eval_metrics.csv"):
        benchmark, activation, seed_dir = csv_path.parts[-4], csv_path.parts[-3], csv_path.parts[-2]
        seed = int(seed_dir.split("_", maxsplit=1)[1])
        frame = pd.read_csv(csv_path)
        frame["benchmark"] = benchmark
        frame["activation"] = activation
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise RuntimeError("No evaluation metrics found to aggregate.")
    return pd.concat(frames, ignore_index=True)


def make_plots(results: pd.DataFrame, output_root: Path) -> List[Path]:
    plot_dir = output_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for metric, ylabel in (("mean_reward", "Evaluation Reward"), ("mean_success", "Normalized Success")):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=results["benchmark"].nunique(),
            figsize=(7 * max(results["benchmark"].nunique(), 1), 5),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, benchmark in zip(axes[0], sorted(results["benchmark"].unique())):
            subset = results[results["benchmark"] == benchmark]
            sns.lineplot(
                data=subset,
                x="timesteps",
                y=metric,
                hue="activation",
                estimator="mean",
                errorbar=("ci", 95),
                marker="o",
                ax=ax,
            )
            ax.set_title(benchmark.replace("_", " "))
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel(ylabel)
        out_path = plot_dir / f"{metric}_learning_curves.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        outputs.append(out_path)
    return outputs


def create_demo_video(
    run_dir: Path,
    benchmark: Benchmark,
    output_root: Path,
    max_frames: int = 300,
) -> Path:
    best_reward_model = run_dir / "best_reward_model.zip"
    model_path = best_reward_model if best_reward_model.exists() else run_dir / "model.zip"
    model = PPO.load(model_path, device="cpu")
    env = make_base_env(benchmark, seed=12345)
    obs, _ = env.reset()
    frames: List[np.ndarray] = []
    done = False
    truncated = False
    steps = 0
    while steps < max_frames and not done and not truncated:
        frame = unwrap_env(env).render("rgb_array")[0]
        frames.append(np.asarray(frame, dtype=np.uint8))
        action, _ = model.predict(raw_obs_to_model_obs(obs), deterministic=True)
        obs, _, done, truncated, _ = env.step(int(action))
        steps += 1
    env.close()

    preview_dir = output_root / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    video_path = preview_dir / f"{benchmark.name}_proposed_demo.mp4"
    imageio.mimsave(video_path, frames, fps=20, macro_block_size=1)
    return video_path


def best_run_for_demo(output_root: Path, benchmark: Benchmark) -> Path:
    best_path: Path | None = None
    best_success = -math.inf
    for summary_path in output_root.glob(f"runs/{benchmark.name}/proposed/seed_*/summary.json"):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary["best_success"] > best_success:
            best_success = float(summary["best_success"])
            best_path = summary_path.parent
    if best_path is None:
        raise RuntimeError(f"No proposed runs found for {benchmark.name}.")
    return best_path


def write_summary(results: pd.DataFrame, plot_paths: Iterable[Path], video_path: Path, output_root: Path) -> Path:
    final_rows = (
        results.sort_values("timesteps")
        .groupby(["benchmark", "activation", "seed"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    best_rows = (
        results.groupby(["benchmark", "activation", "seed"], as_index=False)[["mean_reward", "mean_success"]]
        .max()
        .reset_index(drop=True)
    )
    grouped = (
        final_rows.groupby(["benchmark", "activation"], as_index=False)[["mean_reward", "mean_success"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        for col in grouped.columns.to_flat_index()
    ]
    best_grouped = (
        best_rows.groupby(["benchmark", "activation"], as_index=False)[["mean_reward", "mean_success"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    best_grouped.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        for col in best_grouped.columns.to_flat_index()
    ]
    summary = {
        "plots": [str(path) for path in plot_paths],
        "video": str(video_path),
        "final_metrics": grouped.to_dict(orient="records"),
        "best_metrics": best_grouped.to_dict(orient="records"),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO benchmarks on COOM with baseline and proposed activations.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        type=parse_benchmark,
        default=[parse_benchmark("pitfall-default"), parse_benchmark("hide_and_seek-default")],
        help="Benchmarks in '<scenario>-<task>' format.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1], help="Random seeds to run.")
    parser.add_argument("--timesteps", type=int, default=20_000, help="Training timesteps per run.")
    parser.add_argument("--eval-every", type=int, default=2_500, help="Evaluation interval in timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes per checkpoint.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/coom_activation_validation"),
        help="Artifact output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for benchmark in args.benchmarks:
        for activation in ("gelu", "proposed"):
            for seed in args.seeds:
                train_run(
                    benchmark=benchmark,
                    activation_kind=activation,
                    seed=seed,
                    timesteps=args.timesteps,
                    eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes,
                    output_root=output_root,
                )

    results = aggregate_results(output_root)
    results.to_csv(output_root / "all_eval_metrics.csv", index=False)
    plot_paths = make_plots(results, output_root)
    demo_benchmark = args.benchmarks[0]
    demo_run = best_run_for_demo(output_root, demo_benchmark)
    video_path = create_demo_video(demo_run, demo_benchmark, output_root)
    write_summary(results, plot_paths, video_path, output_root)


if __name__ == "__main__":
    main()
