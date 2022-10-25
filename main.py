import gym
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
from sb3_contrib import TQC
from StockTradingEnv import StockTradingEnv

df = pd.read_csv('./EBAY.csv')
df = df.sort_values('Date')

env = DummyVecEnv([lambda: StockTradingEnv(df)])
# Using SAC model to constitute prediction

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50)

obs = env.reset()
for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

del model

print("=======================================")
print("==========Testing the Next Model=======")
print("=======================================\n")

# Using TQC model to constitute prediction
model = TQC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50)

obs = env.reset()
for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()

# The results differ for each run of the code but
# the TQC model always profits more than the SAC
