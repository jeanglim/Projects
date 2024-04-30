import gymnasium as gym
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

GAME_NAME = "ALE/Pacman-v5"

env_id = f"{GAME_NAME}"
env = make_atari_env(env_id, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model_path = f"{GAME_NAME}_dqn_model"
model = DQN.load(model_path, env=env)

def evaluate_model(model, num_episodes=25, delay = 0.15):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_rewards = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            total_rewards += rewards
            env.render('human')
            time.sleep(delay) 
            if done:
                print(f"Episode {episode+1}: Total Reward: {total_rewards}")
        env.close()

evaluate_model(model)