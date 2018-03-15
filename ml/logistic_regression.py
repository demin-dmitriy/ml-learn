from . gradient_descent import GradientDescentOptimizer

import numpy as np

from math import exp, log


def h(θ):
    return lambda x: 1.0 / (1.0 + exp(-np.dot(θ, x)))

class LogisticRegression:

    def __init__(self, α=0.7, λ=0.0):
        self.α = α
        self.λ = λ

    def fit(self, xs, ys):
        assert len(xs) == len(ys)
        assert len(xs) > 0

        M, N = xs.shape

        def J(θ):
            h_θ = h(θ)
            totalSum = sum(
                (log(1 - h_θ(x)) if y == 0 else log(h_θ(x)))
                for x, y in zip(xs, ys)
            )
            return -totalSum / M

        def dJ(θ):
            h_θ = h(θ)
            return 1 / M * sum(x * (h_θ(x) - y) for x, y in zip(xs, ys))

        return GradientDescentOptimizer(J=J, δJ=dJ, α=self.α, start_θ=[0] * N)
