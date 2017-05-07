
import QuantumComputation as qc
from QuantumComputation import ket

from math import sqrt, cos, pi, acos, asin

n = 4
N = 2**n

def f(b):
    return 1 if b == (0,1,1,0) else 0

zeroes = n*(0,)

def bits(n):
    if n == 0:
        yield ()
    else:
        for b in [0,1]:
            for bs in bits(n-1):
                yield (b,)+bs

# Function to search
def _Uf(t):
    (*b,r) = t
    return ket(*b, (r+f(tuple(b))) % 2)

Uf = qc.QOperator(_Uf)

def _flipzero(t):
    if t[:n] == zeroes:
        return (-1) * ket(*t)
    else:
        return ket(*t)
    
I0 = qc.QOperator(_flipzero)

# Tensor up a Hadamard
Hn = qc.Identity
for i in range(n):
    Hn = Hn * qc.Hadamard.at(i)

# Starting configuration
zero = ket(*zeroes)
ancilla = qc.Hadamard * ket(1)

psi0 = Hn * zero * ancilla

# Grover iteration

alpha = asin(1/sqrt(N))
beta = acos(1/sqrt(N))

nsteps = round(beta / (2*alpha))
napprox = round((pi / 4) * sqrt(N))

gamma = beta - nsteps*(2*alpha)

print("%i Grover iterations ..." % nsteps)
print("Expected amplitude = %f" % (cos(gamma)**2))

state = psi0

for i in range(nsteps):
    state = (-1) * Hn * I0 * Hn * Uf * state

dfinal = state.dist(range(n))

print("Sampling: %s" % str(state.sample(range(n))))