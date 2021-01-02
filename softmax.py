import numpy as np

# y = exp(a_k) / sum(exp(a_i))
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # -cはオーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
