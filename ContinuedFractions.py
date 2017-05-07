# -*- coding: utf-8 -*-
"""
Created on Tue May  2 17:54:51 2017

@author: Dario
"""

# Approximation of positive rationals

def gcd(a, b):
    if b < a:
        a, b = b, a
    
    while a > 0:
        a, b = b % a, a
    
    return b

def cancel(a, b):
    d = gcd(a, b)
    return a//d, b//d

# Compute continued fraction representation
def cf(p, q):
    if p % q == 0:
        return [p // q]
    else:
        a = p // q
        return [a] + cf(*cancel(q,p-a*q))

def cfr(r, n=10):
    a = int(r)
    
    if n == 0 or r == a:
        return [a]
    
    return [a] + cfr(1.0/(r-a),n-1)

# Evaluate continued fraction expressions
def ev(c):
    if len(c) == 0:
        return 0, 1
    elif len(c) == 1:
        return c[0], 1
    else:
        a, (b, c) = c[0], ev(c[1:])
        return cancel(a*b+c, b)

def con(r, n):
    return ev(cfr(r, n))

def convergents(p, q):
    c = cf(p, q)
    for i in range(len(c)):
        truncated = c[:i+1]
        yield ev(truncated)