""" améliorations possibles:
    utiliser une classe de nb complexes !!!
"""

import matplotlib.pyplot as plt 
import numpy as np 
import cv2
import sys
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.image as images

pi = np.pi
cos = np.cos
sin = np.sin

path = '/Users/samyhaffoudhi/Desktop/fourier'
img = sys.argv[1]
N = int(sys.argv[2])
image = cv2.imread(img)
fig, ax = plt.subplots()
ax.axis('off')
ax.set_xlim([0, image.shape[1]])
ax.set_ylim([0, image.shape[0]])
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
        ax.cla()
        ax.axis('off')
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(0, image.shape[0])
        circles = [patches.Circle((0, 0), np.sqrt(a[m][0] ** 2 + a[m][1] ** 2), fill = False) for m in range(len(a))]
        for m in range(2 * N + 1):
            ax.add_patch(circles[m])
        line, = ax.plot([], [], color = "blue")
        points = [ax.plot([], [], ls = "none", marker = "o")] * ( 2 * N + 1)

        def list_points(k):
            l = []
            s = (0, 0)
            for i in range(len(a)):
                u = produit(a[i], exp_i(t[k], i - N))
                s = (s[0] + u[0], s[1] + u[1])
                l.append(s)
            return l

        def animate(k):
            line.set_data(x_plot[:k], y_plot[:k])
            l = list_points(k)
            for m in range(len(a)):
                x_point, y_point = l[m]
                points[m][0].set_data(x_point, y_point)
                circles[m].center = (x_point, y_point)
            return [line] + [points[m][0] for m in range(len(a))] + [circles[m] for m in range(len(a))]
        ani = animation.FuncAnimation(fig = fig, func = animate,
                frames = range(len(x_plot)), interval = 10, repeat = False,  blit = True)

    X.append(event.xdata)
    Y.append(event.ydata)
    ax.scatter(event.xdata, event.ydata, 5, color = "blue")
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

