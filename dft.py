# -*- coding: utf-8 -*-
"""
Created on Sat May  6 15:32:46 2017

@author: Dario
"""

import numpy as np
import matplotlib.pyplot as plt

from math import sqrt, e, pi, floor

N = 10
M = 100
r = 5
A = floor(M/r)
x0 = 2

per = np.zeros(M)
for j in range(0,A):
    per[(x0 + j*r) % M] = (1/sqrt(A))
    
# DFT
dft = np.zeros(M, dtype=np.complex)
omega = e**(2*pi*1j/M)

for c in range(0, M):
    dft[c] = (1/sqrt(M))*sum( per[j]*omega**(c*j) for j in range(0, M) )
    
# Plot

plt.plot(per)
plt.plot(np.absolute(dft))