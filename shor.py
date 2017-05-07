
import QuantumComputation as qc
import ContinuedFractions as cf

from QuantumComputation import ket

from math import sqrt, e, pi

# Shor's algorithm

N = 15
M = 256

a = 2

# Generate superposition 

psi = sum( ket(c,0) for c in range(0,M) ).normalize()

def f(c):
    return pow(a,c,N)

def _Uf(t):
    (c,r) = t
    return ket(c,(r+f(c)) % N)

Uf = qc.QOperator(_Uf)
res = Uf * psi
per = res.measure([1]).discard([1])

# QFT

omega = e**(2*pi*1j/M)

def Fourier(t):
    (a,) = t
    return sum( ((1/sqrt(M)) *omega**(a*b))*ket(b) for b in range(0,M) )

QFT = qc.QOperator(Fourier)

c = QFT * per

# Extract an approximation

def extract():
    (ck,) = c.sample()
    
    for (k,r) in cf.convergents(ck,M):
        if r < N and abs(ck/M - k/r) <= 1/(2*N*N) and f(r)==f(0):
            print("Found period r = %i" % r)
            if r % 2 == 0:
                p = pow(a,r//2,N)
                q1, q2 = cf.gcd(p-1,N), cf.gcd(p+1, N)
                print("Possible factors: %i, %i" % (q1,q2))
                return True
                
    
    print("Algorithm unsuccessful, trying again")
    return False

for i in range(10):
    success = extract()
    if success:
        break