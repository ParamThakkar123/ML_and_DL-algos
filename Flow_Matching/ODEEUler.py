# Simulating an ODE using the Euler method

import numpy as np
import matplotlib.pyplot as plt

def euler_method(f, y0, t0, t_end, h):
    """
    Simulates the ODE dy/dt = f(t, y) using the Euler method.

    Parameters:
    f : function
        The function defining the ODE.
    y0 : float
        The initial condition.
    t0 : float
        The initial time.
    t_end : float
        The end time.
    h : float
        The time step.

    Returns:
    t_values : numpy array
        Array of time values.
    y_values : numpy array
        Array of solution values at each time step.
    """
    n_steps = int((t_end - t0) / h)
    t_values = np.linspace(t0, t_end, n_steps + 1)
    y_values = np.zeros(n_steps + 1)

    y = y0
    y_values[0] = y0

    for i in range(1, n_steps):
        y = y + h * f(t_values[i-1], y)
        y_values[i] = y

    return t_values, y_values

def f(t, y):
    return -2 * y


y0 = 1
t0 = 0
t_end = 5
h = 0.1

t_values, y_values = euler_method(f, y0, t0, t_end, h)

exact = np.exp(-2 * t_values)

# Plotting results
plt.plot(t_values, y_values, 'o-', label="Euler Approximation")
plt.plot(t_values, exact, 'r--', label="Exact Solution")
plt.xlabel("t")
plt.ylabel("y")
plt.legend()
plt.title("Euler Method for ODE dy/dt = -2y")
plt.show()