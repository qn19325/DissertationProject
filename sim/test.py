from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np

unity_env = UnityEnvironment("./envs/FreeAspect", no_graphics=False)
env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
env.reset()
for _ in range(1000):
	action = env.action_space.sample()
	print('action: ', action)
	obs, reward, done, info = env.step(np.abs(2*action))
	print('obs: ', obs)
	print('reward: ', reward)
	print('done: ', done)
	print('info: ', info)
	print('\n\n')

