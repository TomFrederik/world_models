import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np


writer = ani.FFMpegWriter(fps=1)

data = np.arange(10*96*96*3)
data = np.reshape(data/(10*96**2*3), (10*96,96,3))

fig = plt.figure()
im = plt.imshow(np.zeros((96,96,3)), animated = True)
with writer.saving(fig, '/home/tom/Desktop/test.mp4', 640):
    for i in range(10):
        im.set_data(data[i])
        writer.grab_frame()
        print(i)
plt.show()
plt.close()


