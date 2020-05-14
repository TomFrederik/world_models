import matplotlib.pyplot as plt
import matplotlib.animate as ani

writer = ani.FFMpegWriter(fps=30)
im = plt.imshow(np.zeros((96,96,3), animated = True)

data = np.arange(10*96*96*3)
data = np.reshape(data/(10*96**2*3), (10*96,96,3))

fig = plt.figure()
with writer.saving(fig, '/home/tom/Desktop/test.mp4', 10):
    for i in range(10):
        im.set_data(data[i])
        writer.grab_frame()
        print(i)
plt.show()
plt.close()


