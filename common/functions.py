import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))
# y = exp(a_k) / sum(exp(a_i))
# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c) # -cはオーバーフロー対策
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y
def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))



def sigmoid(x):
    return 1 / (1 + np.exp(-x))