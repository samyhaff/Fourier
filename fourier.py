""" améliorations possibles:
    utiliser une classe de nb complexes !!!
"""

import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import sys

pi = np.pi
cos = np.cos
sin = np.sin
path = '/Users/samyhaffoudhi/Desktop/fourier'
img = sys.argv[1]
N = int(sys.argv[2])
fig, ax = plt.subplots()

image = cv2.imread(img)
print(image.shape)
ax.imshow(image[:,:,::-1], extent = [0, image.shape[1], 0, image.shape[0]])

X = []
Y = []
t = np.linspace(0, 2 * pi, 1000)

def produit(z1, z2):
    return (z1[0] * z2[0] - (z1[1] * z2[1]), z1[1] * z2[0] + (z1[0] * z2[1]))

def z(t, n): return (X[int(t * n / (2 * pi))], Y[int(t * n / (2 * pi))])

def I(f, n):
    s1 = 0
    s2 = 0
    h = 2 * pi / n
    for k in range(0, n):
        s1 += f(k * h, n)[0]
        s2 += f(k * h, n)[1]
    return (h * s1, h * s2)

def exp_i(t, k):
    return (cos(k * t), sin(k * t))

def f(k, n):
    def g(t, n): return produit(z(t,n), exp_i(t, k))
    return g

def fourier(n):
    a = [0] * (2 * N + 1)
    for k in range(2 * N + 1):
        x = I(f(k - N, n), n)
        a[k] = (x[0] / (2 * pi), x[1] / (2 * pi))
    return a
        
def Z(t, a):
    s = (0, 0)
    for k in range(len(a)):
        x = produit(a[k], exp_i(t, k - N))
        s = (s[0] + x[0], s[1] + x[1])
    return s

def onclick(event):
    print('%s click: x=%d, y=%d' % 
        ('double' if event.dblclick else 'single', event.xdata, event.ydata))
    if event.dblclick:
        fig.canvas.mpl_disconnect(cid)
        n = len(X)
        print(n)
        print("gathered " + str(n) + " points")
        a = fourier(n)
        x_plot = [Z(u, a)[0] for u in t]
        y_plot = [Z(u, a)[1] for u in t]
        ax.clear()
        ax.plot(x_plot, y_plot, color = "red")

    X.append(event.xdata)
    Y.append(event.ydata)
    ax.scatter(event.xdata, event.ydata, 10, color = "black")
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

