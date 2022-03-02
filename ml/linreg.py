import math
from typing import Type, TypeVar, List, Callable
import numpy as np
from numpy import linalg as LA
import pandas as pd

Term = TypeVar('Term', bound=Callable[[Type[np.ndarray]], float])

def cons() -> Term:
    return lambda f: 1

def id(i: int) -> Term:
    return lambda f: f[i]

def poly(i: int, deg: float) -> Term:
    return lambda f: f[i] ** deg

def exp(i: int) -> Term:
    return lambda f: math.e ** f[i]

def linear(s: int) -> List[Term]:
    return [(lambda f: f[i]) for i in range(s)]

class LinearReg:

    def __init__(self, *terms: Term):
        self.terms = terms
        self.param = np.zeros(len(terms)) # TODO allow other initial parameter
    
    def hypo(self, x: Type[np.ndarray]) -> float:
        # $\vec \theta^T \vec x$
        return np.dot(self.param, x)

    def grad(self, batch: Type[pd.DataFrame], y: Type[pd.Series]) -> float:
        sum = np.zeros(len(self.terms))

        for i, row in batch.iterrows():
            # hstack ensures consistent indexing by the terms because numpy likes horizontal vectors
            features = np.hstack(row.to_numpy()) 
            x = np.array([term(features) for term in self.terms]) # this may cause various error, esp. 
            
            # print the first element for sanity
            if i == 0:
                print("h", self.hypo(x), "y", y[i], "x", x)

            sum += (self.hypo(x) - y[i]) * x

        return sum / batch.shape[0]
    
    def step(self, batch: Type[pd.DataFrame], y: Type[pd.Series], alpha: float) -> float:
        grad = self.grad(batch, y)
        self.param -= grad * alpha
        return LA.norm(grad)

# gradient descent
# TODO: isolate batch and y from graddes' implemention into a more general "data structure"
def graddes(reg: Type[LinearReg], batch: Type[pd.DataFrame], y: Type[pd.Series], alpha: float, eps: float, maxepoch: int) -> bool:
    for i in range(maxepoch):
        print(f"---------\nEpoch #{i}")
        gradsize = reg.step(batch, y, alpha)
        print("Grad size", gradsize)
        print("Param", reg.param)
        if gradsize < eps:
            # converged
            return True
    
    # did not converge
    return False
