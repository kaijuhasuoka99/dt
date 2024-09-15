import numpy as np

def softmax(a):
    c = np.max(a, axis=-1, keepdims=True)
    exp_a = np.exp(a - c)
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)