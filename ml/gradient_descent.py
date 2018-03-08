import matplotlib.pyplot as plt

from utils.itertools_recipes import nth, take


class GradientDescentOptimizer:

    def __init__(self, *, J, δJ, start_θ=None, α=None):
        self._J = J
        self._δJ = δJ
        self.start_θ = start_θ
        self.α = α


    def optimize(self, start_θ=None, α=None):
        if start_θ is None: start_θ = self.start_θ
        assert start_θ is not None

        if α is None: α = self.α
        assert α is not None

        current_θ = start_θ

        while True:
            yield current_θ
            current_θ = current_θ - α * self._δJ(current_θ)


    def nth(self, n, start_θ=None, α=None):
        return nth(self.optimize(start_θ, α), n)


    def plot_progress(self, steps, start_θ=None, α=None):
        history = take(steps, map(self._J, self.optimize(start_θ, α)))

        if α is None: α = self.α

        plt.plot(history, label=f'α = {α}')
        plt.legend()
