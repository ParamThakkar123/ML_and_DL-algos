import numpy as np

class Optimizer:
    def __init__(self, params, learning_rate) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.state = None

    def __repr__(self):
        return f"{self.__class__.__name__}(learning_rate={self.learning_rate})"
    
    def step(self, grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError