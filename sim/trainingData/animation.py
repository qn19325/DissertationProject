import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np

imgs = []

for i in range(33):
    image = Image.open(str(i) + '.png')
    pixels = list(image.getdata())
    array = np.array(pixels)
    shapedArray = array.reshape(64,64)
    imgs.append(shapedArray)

fig, ax = plt.subplots()

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 64)
y = np.linspace(0, 64).reshape(-1, 1)

ims = []
for i in range(33):
    im = ax.imshow(imgs[i], animated=True)
    if i == 0:
        ax.imshow(imgs[0])  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                                repeat_delay=1000)

ani.save("movie.mp4")

plt.show()