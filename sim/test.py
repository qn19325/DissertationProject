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
for _ in range(250):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.abs(2*action))
        arrayObs = np.array(obs[0])
        myData.append((arrayObs, obs[1], index))
        index += 1

data = []
actualData = []
for frame, speed, index in myData:
        data.append((frame, speed))

actualData.append(data[8]) #0
actualData.append(data[15])
actualData.append(data[23])
actualData.append(data[30] )#1
actualData.append(data[38])
actualData.append(data[45])
actualData.append(data[53] )#2
actualData.append(data[60])
actualData.append(data[68])
actualData.append(data[75] )#3
actualData.append(data[83])
actualData.append(data[90])
actualData.append(data[98] )#4
actualData.append(data[105])
actualData.append(data[113])
actualData.append(data[120] )#5
actualData.append(data[128])
actualData.append(data[135])
actualData.append(data[143] )#6
actualData.append(data[150])
actualData.append(data[158])
actualData.append(data[165] )#7
actualData.append(data[173])
actualData.append(data[180])
actualData.append(data[188] )#8
actualData.append(data[195])
actualData.append(data[203])
actualData.append(data[210] )#9
actualData.append(data[218])
actualData.append(data[225])
actualData.append(data[233] )#10
actualData.append(data[240])
actualData.append(data[248])
# print(actualData)

for x in range(len(actualData)):
        pixels = []
        for i in range(64):
                row = actualData[x][0][i].flatten()
                pixels.append(row)
        array = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(str(x) + '.png')


# with open('data.npy', 'wb') as f:
#         np.save(f, actualData, allow_pickle=True)
# with open('data.npy', 'rb') as f:
#         a = np.load(f, allow_pickle=True)
#         print(a)
