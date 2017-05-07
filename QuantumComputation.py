
from math import sqrt
from random import random 

# Helper functions
def sampleDist(dist):
    r = random()
    acc = 0
    
    for (k,p) in dist:
        acc += p
        if acc >= r:
            return k

def extract(t, indices):
    t0 = []
    for i in indices:
        t0.append(t[i])
    return tuple(t0)

def insert(s, t, indices):
    l = list(t)
    c = 0
    for i in indices:
        l[i] = s[c]
        c += 1
    return tuple(l)

def conj(a):
    if isinstance(a, complex):
        return a.conjugate()
    else:
        return a

# A quantum state (tensor product)
class QState():
    
    def __simplify(coefflist):
        summands = {}
        for (x,c) in coefflist:
            if c != 0:
                if x not in summands:
                    summands[x] = c
                else:
                    summands[x] += c
        return list( (x,c) for (x,c) in summands.items() if c != 0 )
    
    def summands(self):
        return self.__summands
    
    def __init__(self, summands):
        self.__summands = QState.__simplify(summands)
        
    def __str__(self):
        return str(self.__summands)
    
    def __repr__(self):
        return str(self)
    
    # Arithmetic
    
    def __mul__(self, b): # self*b
        if isinstance(b, QState):
            tensor = ( (x+y,c*d) for (x,c) in self.summands() for (y,d) in b.summands() )
            return QState(tensor)
        else:
            return QState((x,b*c) for (x,c) in self.summands())
        
    def __rmul__(self, a): # a*self
        return QState((x,a*c) for (x,c) in self.summands())
    
    def __add__(self, b): # self + b
        if isinstance(b, QState):
            return QState(self.summands() + b.summands())
        else: # sum() requires a + 0
            return self
    
    def __radd__(self, a): # a + self
        return self.__add__(a)
    
    def __sub__(self, b):
        return self + (-1 * b)
    
    def dot(self, b): # inefficient af ;)
        return sum( conj(c)*d for (x,c) in self.summands() for (y,d) in b.summands() if x==y )
    
    def norm(self):
        return sqrt(sum( abs(c*c) for (x,c) in self.summands() ))
    
    def normalize(self):
        return (1/self.norm())*self
    
    # Measurement
    def __defaultIndices(self):
        (x,c) = self.summands()[0]
        return range(len(x))
    
    def dist(self, indices=None):
        indices = indices or self.__defaultIndices()
        
        d = {}
        for (x,c) in self.summands():
            t = extract(x,indices)
            if t not in d:
                d[t] = abs(c*c)
            else:
                d[t] += abs(c*c)
        return d
    
    def measure(self, indices=None):
        indices = indices or self.__defaultIndices()
        
        result = self.sample(indices)
        collapse = QState( (x,c) for (x,c) in self.summands() if extract(x,indices) == result )
        return (1/collapse.norm()) * collapse 
    
    def sample(self, indices=None):
        indices = indices or self.__defaultIndices()
        return sampleDist(self.dist(indices).items())
    
    def discard(self, indices):
        post = self.measure(indices)
        remaining = [ i for i in post.__defaultIndices() if i not in indices ]
        return QState( (extract(x, remaining),c) for (x,c) in post.summands() )

def ket(*c):
    return QState([(c,1)])

# Quantum unitary operators
class QOperator():
    def __init__(self, F):
        self.__F = F
        
    def applyOnTensor(self, t):
        return self.__F(t)
        
    def apply(self, state):        
        return sum( c*self.applyOnTensor(x) for (x,c) in state.summands() )

    def after(self, B):
        return QOperator(lambda t : self.apply(B.applyOnTensor(t)))
    
    def __rmul__(self, a): # a*self
        if isinstance(a, QOperator):
            return a.after(self)
        else:
            return QOperator(lambda t : a * self.applyOnTensor(t))
    
    def __mul__(self, b): # self*b
        if isinstance(b, QOperator):
            return self.after(b)
        else:
            return self.apply(b)
    
    def at(self, *indices):
        def apply(tfull):
            t = extract(tfull, indices)
            ret = self.applyOnTensor(t)
            return QState( (insert(x, tfull, indices), c) for (x,c) in ret.summands() )
        return QOperator(apply)
    
# Define some operators
def __Hadamard(t):
    (b,) = t
    
    if b == 0:
        return (1/sqrt(2))*(ket(0) + ket(1))
    elif b == 1:
        return (1/sqrt(2))*(ket(0) - ket(1))
    
Hadamard = QOperator(__Hadamard)
Identity = QOperator(lambda t: ket(*t))