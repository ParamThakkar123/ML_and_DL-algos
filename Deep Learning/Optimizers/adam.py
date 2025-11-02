from Optimizers.optimizer import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, params, learning_rate=0.001, beta_1=0.9, beta_2=0.95, epsilon=1e-8, maximise=False, amsgrad=False) -> None:
        super().__init__(params, learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = np.zeros_like(params)
        self.v = np.zeros_like(params)
        self.t = 0
        self.maximise = maximise
        self.amsgrad = amsgrad

    def step(self, grads: np.ndarray) -> np.ndarray:
        if self.maximise:
            grads = -grads

        self.t += 1
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grads
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grads ** 2)
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        if self.amsgrad:
            if not hasattr(self, 'v_max'):
                self.v_max = np.zeros_like(self.v)
            self.v_max = np.maximum(self.v_max, self.v)
            v_hat = self.v_max / (1 - self.beta_2 ** self.t)
        else:
            v_hat = self.v / (1 - self.beta_2 ** self.t)

        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.params -= update
        return self.params