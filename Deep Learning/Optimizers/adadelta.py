from Optimizers.optimizer import Optimizer
import numpy as np

class Adadelta(Optimizer):
    def __init__(self, params, learning_rate=1.0, rho=0.95, weight_decay=0.0, eps=1e-6):
        super().__init__(params, learning_rate)

        self.params = list(params)
        self.learning_rate = learning_rate
        self.rho = rho
        self.weight_decay = weight_decay
        self.eps = eps

        self.square_avg = [np.zeros_like(p) for p  in self.params]
        self.acc_delta = [np.zeros_like(p) for p in self.params]

    def step(self, grads: np.ndarray) -> np.ndarray:
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g.copy()    
            if self.weight_decay != 0:
                g = g + self.weight_decay * p
            self.square_avg[i] = self.rho * self.square_avg[i] + (1 - self.rho) * (g ** 2)
            std = np.sqrt(self.square_avg[i] + self.eps) / np.sqrt(self.acc_delta[i] + self.eps) * g
            p = p - self.learning_rate * std
            self.acc_delta[i] = self.rho * self.acc_delta[i] + (1 - self.rho) * (std ** 2)
            self.params[i] = p
        return self.params
