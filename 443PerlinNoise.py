import random
import math
import numpy
import matplotlib.pyplot as plt

random.seed() # Optional seed 

n = 40 # Number of random 1D vectors
t = 3 # Number of samples between each vectors (corners)
interpolationStep = 8 # Number of interpolated points between each sample

a = [(random.random()-0.5)*2 for i in range(n)]
b = []

for i in range(int((len(a)-1) * t)): # Sample points
    pos = i/t
    start = int(pos)
    end = start + 1
    
    startVect = (a[start], 0)
    endVect = (a[end], 0)
    candToStart = (start - pos, 0)
    candToEnd = (end - pos, 0)
    if (abs(candToStart[0]) > abs(candToEnd[0])):
        candValue = numpy.dot(candToEnd, endVect)
    else:
        candValue = numpy.dot(candToStart, startVect)
    b.append(candValue)    

c = []

""" Cosine interpolation
cx = numpy.linspace(0,len(b), num=len(b)*interpolationStep, endpoint=True)
for i in cx:
    pos = i/interpolationStep
    start = int(pos)
    end = start + 1
    mu = pos - start
    
    mu2 = (1-numpy.cos(mu*math.pi))/2
    c.append(b[start]*(1-mu2)+b[end]*mu2)
"""

cx = numpy.linspace(2,len(b)-2, num=len(b)*interpolationStep, endpoint=True)
for i in cx: # Cubic interpolation
    pos = i/interpolationStep
    start = int(pos)
    end = start + 1
    mu = pos - start
    mu2 = mu**2
    
    y0,y1,y2,y3 = b[start - 1], b[start], b[end], b[end + 1]
    a0 = y3 - y2 - y0 + y1
    a1 = y0 - y1 - a0
    a2 = y2 - y0
    a3 = y1

    c.append(a0*mu*mu2+a1*mu2+a2*mu+a3)

plt.plot(cx,c)






















