from Optimizers.optimizer import Optimizer
import numpy as np

class Adafactor(Optimizer):
    def __init__(self, params, learning_rate=1e-2, beta_2=0.999, eps_1=1e-30, eps_2=1e-3, weight_decay=0.0, clipping_threshold=1.0, maximise=False):
        super().__init__(params, learning_rate)
        self.beta_2 = beta_2
        self.eps_1 = eps_1
        self.eps_2 = eps_2
        self.weight_decay = weight_decay
        self.clipping_threshold = clipping_threshold
        self.row_moments = [np.zeros((p.shape[0], 1)) for p in self.params]
        self.col_moments = [np.zeros((1, p.shape[1])) for p in self.params]
        self.vectors = [np.zeros_like(p) for p in self.params]
        self.maximise = maximise
        self.t = 0

    def step(self, grads: np.ndarray) -> np.ndarray:
        self.t += 1
        lr_t = min(self.learning_rate, 1.0 / np.sqrt(self.t)) if self.learning_rate is not None else 1.0 / np.sqrt(self.t)

        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = g.copy()
            if self.maximise:
                g = -g
            if self.weight_decay != 0:
                p = p - self.weight_decay * lr_t * p

            grad_sq = g * g + self.eps_1

            if p.ndim >= 2:
                row_mean = np.mean(grad_sq, axis=1, keepdims=True)
                col_mean = np.mean(grad_sq, axis=0, keepdims=True)

                if self.row_moments[i] is None or self.col_moments[i] is None:
                    self.row_moments[i] = np.zeros_like(row_mean)
                    self.col_moments[i] = np.zeros_like(col_mean)
                
                self.row_moments[i] = self.beta_2 * self.row_moments[i] + (1 - self.beta_2) * row_mean
                self.col_moments[i] = self.beta_2 * self.col_moments[i] + (1 - self.beta_2) * col_mean

                denom = np.maximum(self.eps_1, np.mean(self.row_moments[i]))
                approx_v = (self.row_moments[i] @ self.col_moments[i]) / denom
                v = approx_v
            else:
                self.vectors[i] = self.beta_2 * self.vectors[i] + (1 - self.beta_2) * grad_sq
                v = self.vectors[i]

            update = g / (np.sqrt(v) + self.eps_2)

            if self.clipping_threshold is not None and self.clipping_threshold > 0.0:
                rms_update = np.sqrt(np.mean(update * update))
                clip_denom = max(1.0, rms_update / self.clipping_threshold)
                update = update / clip_denom

            p = p - lr_t * update
            self.params[i] = p

        return self.params