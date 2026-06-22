import argparse
import csv
import json
import os
import sys
from pathlib import Path
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "footsies_gym"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers import FootsiesActionCombinationsDiscretized, FootsiesNormalized, FootsiesStatistics


def make_env(game_path, render_mode, sync_mode, fast_forward, fast_forward_speed, log_file, frame_delay, skip_instancing):
    def _init():
        env = FootsiesEnv(
            game_path=game_path,
            render_mode=render_mode,
            sync_mode=sync_mode,
            vs_player=False,
            fast_forward=fast_forward,
            fast_forward_speed=fast_forward_speed,
            log_file=log_file,
            log_file_overwrite=True,
            frame_delay=frame_delay,
            skip_instancing=skip_instancing,
        )

        env = FootsiesNormalized(env)
        env = FootsiesStatistics(env)
        env = FootsiesActionCombinationsDiscretized(env)
        env = Monitor(env)
        return env

    return _init


def resolve_repo_path(path_like: str) -> str:
    path = Path(path_like)
    return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())


class PolicyEntropyCSVCallback(BaseCallback):
    """Logs policy entropy H(s) = -sum_a pi(a|s) log pi(a|s) to CSV at each training step."""

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = csv_path
        self._file_handle = None
        self._writer = None

    def _on_training_start(self) -> None:
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        self._file_handle = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file_handle)
        self._writer.writerow(["timesteps", "env_index", "entropy"])

    def _on_step(self) -> bool:
        if self._writer is None:
            return True

        obs = self.locals.get("new_obs")
        if obs is None:
            return True

        with torch.no_grad():
            obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
            distribution = self.model.policy.get_distribution(obs_tensor)
            entropy = distribution.distribution.entropy()

            if entropy.dim() > 1:
                entropy = entropy.sum(dim=-1)

            entropy_values = entropy.detach().cpu().numpy().tolist()

        for env_idx, entropy_value in enumerate(entropy_values):
            self._writer.writerow([self.num_timesteps, env_idx, float(entropy_value)])

        self._file_handle.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._writer = None


def find_statistics_wrapper(env) -> FootsiesStatistics:
    current = env
    while current is not None:
        if isinstance(current, FootsiesStatistics):
            return current
        current = getattr(current, "env", None)

    raise RuntimeError("FootsiesStatistics wrapper was not found in the environment stack")


def aggregate_statistics(vec_env: DummyVecEnv) -> dict:
    summaries = [find_statistics_wrapper(env).summary() for env in vec_env.envs]
    total_episodes = sum(summary["episodes"] for summary in summaries)

    if total_episodes == 0:
        return {
            "episodes": 0,
            "wins": 0,
            "winrate": 0.0,
            "special_moves_total": 0,
            "special_moves_average": 0.0,
            "special_moves_from_neutral_total": 0,
            "special_moves_from_neutral_average": 0.0,
            "combo_stance_steps_total": 0,
            "combo_stance_steps_average": 0.0,
            "combo_armed_steps_total": 0,
            "combo_armed_steps_average": 0.0,
            "combo_toggles_total": 0,
            "combo_toggles_average": 0.0,
            "combo_followups_queued_total": 0,
            "combo_followups_queued_average": 0.0,
        }

    totals = {
        "episodes": total_episodes,
        "wins": sum(summary["wins"] for summary in summaries),
        "special_moves_total": sum(summary["special_moves_total"] for summary in summaries),
        "special_moves_from_neutral_total": sum(summary["special_moves_from_neutral_total"] for summary in summaries),
        "combo_stance_steps_total": sum(summary["combo_stance_steps_total"] for summary in summaries),
        "combo_armed_steps_total": sum(summary["combo_armed_steps_total"] for summary in summaries),
        "combo_toggles_total": sum(summary["combo_toggles_total"] for summary in summaries),
        "combo_followups_queued_total": sum(summary["combo_followups_queued_total"] for summary in summaries),
    }

    totals["winrate"] = totals["wins"] / total_episodes
    totals["special_moves_average"] = totals["special_moves_total"] / total_episodes
    totals["special_moves_from_neutral_average"] = totals["special_moves_from_neutral_total"] / total_episodes
    totals["combo_stance_steps_average"] = totals["combo_stance_steps_total"] / total_episodes
    totals["combo_armed_steps_average"] = totals["combo_armed_steps_total"] / total_episodes
    totals["combo_toggles_average"] = totals["combo_toggles_total"] / total_episodes
    totals["combo_followups_queued_average"] = totals["combo_followups_queued_total"] / total_episodes

    return totals


def format_statistics(summary: dict) -> str:
    lines = [
        "Training statistics",
        f" Episodes: {summary['episodes']}",
        " Results",
        f"  Wins: {summary['wins']}",
        f"  Winrate: {summary['winrate']:.3f}",
        " Special moves",
        f"  Average: {summary['special_moves_average']}",
        f"  Total: {summary['special_moves_total']}",
        " Special moves from neutral",
        f"  Average: {summary['special_moves_from_neutral_average']}",
        f"  Total: {summary['special_moves_from_neutral_total']}",
        " Combo stance",
        f"  Steps average: {summary['combo_stance_steps_average']}",
        f"  Steps total: {summary['combo_stance_steps_total']}",
        f"  Toggles average: {summary['combo_toggles_average']}",
        f"  Toggles total: {summary['combo_toggles_total']}",
        " Combo armed",
        f"  Steps average: {summary['combo_armed_steps_average']}",
        f"  Steps total: {summary['combo_armed_steps_total']}",
        " Combo follow-ups queued",
        f"  Average: {summary['combo_followups_queued_average']}",
        f"  Total: {summary['combo_followups_queued_total']}",
    ]
    return "\n".join(lines)


def write_statistics(summary: dict, text_path: str, json_path: str):
    text_report = format_statistics(summary)
    print(text_report, flush=True)

    text_dir = os.path.dirname(text_path)
    if text_dir:
        os.makedirs(text_dir, exist_ok=True)
    with open(text_path, "w", encoding="utf-8") as handle:
        handle.write(text_report + "\n")

    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on Footsies Unity environment")
    parser.add_argument("--game-path", default="Build/FOOTSIES.exe", help="Path to the FOOTSIES.exe Unity build")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save-path", default="Agents-Test/ppo_footsies", help="Path prefix to save the trained model")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs (DummyVecEnv)")
    parser.add_argument("--render", action="store_true", help="Enable rendering (human). Default: off")
    parser.add_argument("--fast-forward", action="store_true", help="Enable fast forward in the env")
    parser.add_argument("--fast-forward-speed", type=float, default=6.0, help="Fast forward speed multiplier")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity for SB3")
    parser.add_argument("--tensorboard-log", default=None, help="Tensorboard log dir")
    parser.add_argument("--stats-text-path", default="Agents-Test/latest_training_stats.txt", help="Path to write the human-readable training statistics report")
    parser.add_argument("--stats-json-path", default="Agents-Test/latest_training_stats.json", help="Path to write the JSON training statistics report")
    parser.add_argument("--entropy-csv-path", default="Agents-Test/policy_entropy.csv", help="Path to write per-timestep policy entropy CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.game_path = resolve_repo_path(args.game_path)
    args.save_path = resolve_repo_path(args.save_path)
    args.stats_text_path = resolve_repo_path(args.stats_text_path)
    args.stats_json_path = resolve_repo_path(args.stats_json_path)
    args.entropy_csv_path = resolve_repo_path(args.entropy_csv_path)

    if not os.path.isfile(args.game_path):
        raise FileNotFoundError(f"Unity build not found at '{args.game_path}'. Build your game or provide --game-path.")

    render_mode = "human" if args.render else None

    env_fns = [
        make_env(
            game_path=args.game_path,
            render_mode=render_mode,
            sync_mode="synced_non_blocking",
            fast_forward=args.fast_forward,
            fast_forward_speed=args.fast_forward_speed,
            log_file=str((REPO_ROOT / "out.log").resolve()),
            frame_delay=0,
            skip_instancing=False,
        )
        for _ in range(max(1, args.n_envs))
    ]

    env = DummyVecEnv(env_fns)

    model = PPO("MultiInputPolicy", env, verbose=args.verbose, tensorboard_log=args.tensorboard_log)
    entropy_callback = PolicyEntropyCSVCallback(csv_path=args.entropy_csv_path)

    try:
        model.learn(total_timesteps=args.timesteps, callback=entropy_callback)
    except KeyboardInterrupt:
        print("Training interrupted by user — saving model before exit...")
    finally:
        try:
            save_dir = os.path.dirname(args.save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            model.save(args.save_path)
            print(f"Model saved to {args.save_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
        try:
            write_statistics(
                aggregate_statistics(env),
                args.stats_text_path,
                args.stats_json_path,
            )
        except Exception as e:
            print(f"Failed to collect training statistics: {e}")
        env.close()