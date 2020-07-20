import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import sys

pi = np.pi
cos = np.cos
sin = np.sin
path = '/Users/samyhaffoudhi/Desktop/fourier'
img = sys.argv[1]
fig, ax = plt.subplots()

image = cv2.imread(img)
print(image.shape)
ax.imshow(image[:,:,::-1], extent = [0, image.shape[1], 0, image.shape[0]])

x = []
y = []


def onclick(event):
    print('%s click: button=%d, x=%d, y=%d' % 
        ('double' if event.dblclick else 'single', event.button,
        event.xdata, event.ydata))
    if event.dblclick:
        fig.canvas.mpl_disconnect(cid)
        print("Done gathering points")
    x.append(event.xdata)
    y.append(event.ydata)
    ax.scatter(event.xdata, event.ydata, 10, color = "black")
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

