"""
Discrete module containing the discrete derivative function.
"""

def diff(t, x):
    if len(t) != len(x):
        raise ValueError("t and x must have equal length")
    return [(x[k] - x[k-1]) / (t[k] - t[k-1]) for k in range(1, len(t))]  # v(t) = dx/dt
