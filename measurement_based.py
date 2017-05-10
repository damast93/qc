

import QuantumComputation as qc
from QuantumComputation import ket

from math import sqrt, e, pi

# Basis operators
H = qc.Hadamard

# X=NOT, swapping 0,1
def _X(t):
    (a,) = t
    if a == 0:
        return ket(1)
    else:
        return ket(0)

X = qc.QOperator(_X)

# Z = inversion at 1
def _Z(t):
    (a,) = t
    if a == 0:
        return ket(0)
    else:
        return (-1) * ket(1)
    
Z = qc.QOperator(_Z)

# E = CZ entanglement gate
def _E(t):
    (a,b) = t
    if a == 0:
        return ket(a,b)
    else:
        return ket(a) * (Z * ket(b))

E = qc.QOperator(_E)

# J-basis
def _J(alpha, t):
    (a,) = t
    if a == 0:
        return (1/sqrt(2))*(ket(0) + ket(1))
    else:
        return (1/sqrt(2))*( e**(1j*alpha)*ket(0) - e**(1j*alpha)*ket(1) )

def _Jinv(alpha, t):
    (a,) = t
    if a == 0:
        return (1/sqrt(2))*(ket(0) + e**(-1j*alpha)*ket(1))
    else:
        return (1/sqrt(2))*(ket(0) - e**(-1j*alpha)*ket(1) )

def J(alpha):
    return qc.QOperator(lambda t: _J(alpha, t))

def Jinv(alpha):
    return qc.QOperator(lambda t: _Jinv(alpha, t))

def vplus(alpha):
    return (1/sqrt(2))*(ket(0) + e**(-1j*alpha)*ket(1))

def vminus(alpha):
    return (1/sqrt(2))*(ket(0) - e**(-1j*alpha)*ket(1))

# [vplus(alpha), vminus(alpha)] is an ONB B(alpha)
# J(alpha) is the base change back to [0, 1], i.e.
# J(alpha) * vplus(alpha) == ket(0)
# J(alpha) * vminus(alpha) == ket(1)

plus = vplus(0)

# Universality of {J,E}
CX = J(0).at(1) * E * J(0).at(1)
SWAP = CX * CX.at(1,0) * CX

# J-process

alpha = 2
beta = 0.53
psi = 0.6 * ket(0) + 0.8*ket(1)

graph = E.at(0,1) * E.at(1,2) * (psi * plus * plus)

def M(s, i, alpha):
    pre = J(alpha).at(i) * s
    post = pre.measure([i])
    (s,) = post.sample([i])
    ret = Jinv(alpha).at(i) * post
    return s, ret

s1, stage1 = M(graph, 0, alpha)
s2, stage2 = M(stage1, 1, (-1)**s1 * beta)

postprocess = X**s2 * stage2.discard([0,1])

aim = J(beta) * J(alpha) * psi

print(aim.dist())
print(postprocess.dist())