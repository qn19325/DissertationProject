import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import matplotlib.animation as animation

dims = "64x64" # size of images 
csvFileName = ""

images = []
labels = []

labels_df = pd.read_csv('{}.csv'.format(dims))

for i, l in np.array(labels_df):
    img = torchvision.io.read_image('{}/{}'.format(dims, i))
    images.append(img.squeeze().numpy())
    labels.append(l)
images = np.array(images)
labels = np.array(labels)
print(images.shape, labels.shape)

images_plot = []
fig, ax = plt.subplots()
imgs = []
for i in range(len(images)):
    im = ax.imshow(images[i], animated=True)
    if i == 0:
        ax.imshow(images[i])  # show an initial one first
    t = ax.annotate(labels[i],(1,0.), fontsize=25, color='red') # add text
    imgs.append([im, t])

ani = animation.ArtistAnimation(fig, imgs, interval=500, blit=True, repeat_delay=1000) 
ani.save("{}.mp4".format(dims))

plt.show()