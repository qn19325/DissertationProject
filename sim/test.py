from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

unity_env = UnityEnvironment("./envs/64x64", no_graphics=False)
env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)
env.reset()
myData = []
index = 0
for _ in range(252):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.abs(2*action))
        arrayObs = np.array(obs[0])
        myData.append((arrayObs, obs[1], index))
        index += 1

# data = []
# for frame, speed, index in myData:
#         data.append((frame, speed))

for x in range(len(myData)):
        pixels = []
        for i in range(64):
                row = myData[x][0][i].flatten()
                pixels.append(row)
                index = myData[x][1]
        array = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save('training/' + str(x) + '.png')

with open('data.npy', 'wb') as f:
        np.save(f, myData, allow_pickle=True)
with open('data.npy', 'rb') as f:
        a = np.load(f, allow_pickle=True)
        print(a)
