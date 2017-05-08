import QuantumComputation as qc
from QuantumComputation import ket

from math import e, pi

# Converting from bits to numbers
def bits(n):
    return tuple(map(int, "{0:04b}".format(n)))

def num(b):
    return int("".join(map(str,b)), base=2)

# Set up our operator, prepare an eigenstate psi
psi = ket(0)
theta = (13/16) # 0.1101

def _u(t):
    (b,) = t
    if b == 0:
        return e**(2*pi*1j*theta) * ket(0)
    else:
        return ket(1)

def U(n):
    op = qc.Identity
    for i in range(n):
        op = op * qc.QOperator(_u)
    return op

def CU(n):
    def apply(t):
        (c,x) = t
        if c == 0:
            return ket(c,x)
        else:
            return ket(c) * (U(n) * ket(x))
    return qc.QOperator(apply)
    

# Phase estimation
H = qc.Hadamard

p = H * ket(0)
start = p * p * p * p * psi

post = CU(2**3).at(0,4) * start
post = CU(2**2).at(1,4) * post
post = CU(2**1).at(2,4) * post
post = CU(2**0).at(3,4) * post

# IQFT
omega = e**(-2*pi*1j/16)

def Fourier(t):
    a = num(t[0:4])
    return sum( ((1/4) * omega**(a*b))*ket(*bits(b)) for b in range(0,16) )

IQFT = qc.QOperator(Fourier)

app = IQFT.at(0,1,2,3) * post
approx = app.sample([0,1,2,3])
thetaPrime = num(approx)/16