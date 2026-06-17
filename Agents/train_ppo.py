import gymnasium as gym
from stable_baselines3 import PPO
from footsies_gym import FootsiesEnv, FootsiesNormalized, FootsiesActionCombinationsDiscretized

env = FootsiesEnv(
    game_path="Build/FOOTSIES.exe",
    render_mode="human",
    sync_mode="synced_non_blocking",
    vs_player=False,
    fast_forward=True,
    fast_forward_speed=6.0,
    log_file="out.log",
    log_file_overwrite=True,
    frame_delay=0,
    skip_instancing=False,
)

env = FootsiesNormalized(env)
env = FootsiesActionCombinationsDiscretized(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("ppo_footsies")
env.close()