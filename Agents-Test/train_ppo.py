import os
import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from footsies_gym.envs.footsies import FootsiesEnv
from footsies_gym.wrappers import FootsiesNormalized, FootsiesActionCombinationsDiscretized


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
        env = FootsiesActionCombinationsDiscretized(env)
        env = Monitor(env)
        return env

    return _init


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

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
            log_file="out.log",
            frame_delay=0,
            skip_instancing=False,
        )
        for _ in range(max(1, args.n_envs))
    ]

    env = DummyVecEnv(env_fns)

    model = PPO("MultiInputPolicy", env, verbose=args.verbose, tensorboard_log=args.tensorboard_log)

    try:
        model.learn(total_timesteps=args.timesteps)
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
        env.close()