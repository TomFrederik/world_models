import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np

writer = ani.FFMpegWriter(fps=1)


data = np.arange(10*96*96*3)
data = np.reshape(data/(10*96**2*3), (10*96,96,3))

fig = plt.figure()
im = plt.imshow(np.zeros((96,96,3)), animated = True)
with writer.saving(fig, '/home/tom/Desktop/test_ani.mp4', dpi=480):
    for i in range(10):
        im.set_data(data[i])
        if i>4:
            d = np.zeros((96,96,3))
            d[:,:,0] = np.ones((96,96))
            im.set_data(d)
        writer.grab_frame()
        print(i)

plt.savefig('/home/tom/Desktop/test_pic_0.pdf')

im1 = plt.imshow(d)
plt.savefig('/home/tom/Desktop/test_pic_1.pdf')

d_1 = d.copy()
d_1[:,:,1] = np.ones((96,96))

im1.set_data(d_1)
plt.savefig('/home/tom/Desktop/test_pic_2.pdf')
