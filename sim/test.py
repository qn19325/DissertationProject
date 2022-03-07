from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
import pickle as pkl

unity_env = UnityEnvironment("./envs/64x64", no_graphics=False)
env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)
env.reset()
myData = []
index = 0
for _ in range(250):
	action = env.action_space.sample()
#	print('action: ', action)
	obs, reward, done, info = env.step(np.abs(2*action))
	myData.append((obs[0], obs[1], index))
	index += 1

data = {}
actualData = {}
for frame, speed, index in myData:
        data[index] = (frame, speed)
        #print(frame) 
actualData[1] = data[8] #0
actualData[2] = data[15]
actualData[3] = data[23]
actualData[4] = data[30] #1
actualData[5] = data[38]
actualData[6] = data[45]
actualData[7] = data[53] #2
actualData[8] = data[60]
actualData[9] = data[68]
actualData[10] = data[75] #3
actualData[11] = data[83]
actualData[12] = data[90]
actualData[13] = data[98] #4
actualData[14] = data[105]
actualData[15] = data[113]
actualData[16] = data[120] #5
actualData[17] = data[128]
actualData[18] = data[135]
actualData[19] = data[143] #6
actualData[20] = data[150]
actualData[21] = data[158]
actualData[22] = data[165] #7
actualData[23] = data[173]
actualData[24] = data[180]
actualData[25] = data[188] #8
actualData[26] = data[195]
actualData[27] = data[203]
actualData[28] = data[210] #9
actualData[29] = data[218]
actualData[30] = data[225]
actualData[31] = data[233] #10
actualData[32] = data[240]
actualData[33] = data[248]


with open('data.npy', 'wb') as f:
        np.save(f, actualData, allow_pickle=True)
with open('data.npy', 'rb') as f:
        a = np.load(f, allow_pickle=True)
        print(a)
