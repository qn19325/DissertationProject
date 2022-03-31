import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import matplotlib.animation as animation

dims="256x256"
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
## animate and save mp4
images_plot = []
fig, ax = plt.subplots()
ims = []
for i in range(len(images)):
    im = ax.imshow(images[i], animated=True)
    if i == 0:
        ax.imshow(images[i])  # show an initial one first
    t = ax.annotate(labels[i],(1,0.), fontsize=25, color='red') # add text
    ims.append([im, t])
ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True,
                                repeat_delay=1000) # inteval: delay between each frames in ms
ani.save("{}.mp4".format(dims))
#
# or
#
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.show()