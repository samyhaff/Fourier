"""given an image, returns the parametric equation""" 

import cv2
import numpy
import os # demander le nom de l'image en argument 

path = '/Users/samyhaffoudhi/Desktop/fourier/'
image =  'tux.png'

img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

p = 100 
n = img.shape[0]
m = img.shape[1]
N = 10
a = [0] * N
b = [0] * N 
x = []
y = []
t = []

c = 0
for i in range(n):
    for j in range(m):
        c += 1
        if img[i][j] != 0:
            t.append(c)
            x.append(abs(j - (m / 2)))
            y.append(abs(i - (n / 2)))

l = len(t)
x = [2 * np.pi * x / l for x in t]

def x(t): 
    s = 0
    for i in range(l):
        p = 1
        for j in range(l):
            if j != i:
                p *= (t - t[i]) / (t[i] - t[j])
        s += x[i] * p
        return s
        
def y(t): 
    s = 0
    for i in range(l):
        p = 1
        for j in range(l):
            if j != i:
                p *= (t - t[i]) / (t[i] - t[j])
        s += y[i] * p
        return s

def integrer(f):
    s = 0
    h = 2 * np.pi / p
    for k in range(p + 1):
        s += f(k * 2 * np.pi / p)
    return h * s 

a[0] = integrer(x) / (2 * np.pi)
b[0] = integrer(y) / (2 * np.pi)

for k in range(1, N + 1):
    a[k] = integrer(lambda t: x(t) * np.cos(k * t)) / np.pi
    b[k] =  integrer(lambda t: y(t) * np.sin(k * t)) / np.pi
   















