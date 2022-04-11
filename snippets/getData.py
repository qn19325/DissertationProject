from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
import numpy as np
from PIL import Image
import csv

envPath = '' # where executbale environment is stored
numOfObservations = 252 # number of observations to make
imageSize = '64x64' # image sizes
csvFile = '' # csv file yo write to 

unity_env = UnityEnvironment("./" + envPath, no_graphics=False)
env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=True)
env.reset()
myData = []
index = 0
for _ in range(numOfObservations):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(np.abs(2*action))
        arrayObs = np.array(obs[0])
        myData.append((arrayObs, np.array(obs[1]).item(), index))
        index += 1

counter = 0
for x in range(len(myData)):
        pixels = []
        speed = myData[x][1] / 10
        for i in range(64):
                row = myData[x][0][i].flatten()
                pixels.append(row)
                index = myData[x][1]
        array = np.array(pixels, dtype=np.uint8)
        new_image = Image.fromarray(array)
        new_image.save(imageSize + '/' + str(counter) + '.png')
        counter = counter + 1

with open(csvFile, 'w', newline='') as file:
        for i in range(len(myData)):
                writer = csv.writer(file)
                speed = myData[i][1]
                path = str(i) + ".png"
                writer.writerow([path, speed])