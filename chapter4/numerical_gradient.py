import numpy as np

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # xと同じ形状の配列を生成

    for idx in range(x.size):
        print(idx)
        print(x)
        tmp_val = x[idx] # 3.0
        # f(x+h)の計算
        x[idx] = tmp_val + h # x0 = 3.0+h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h # x0 = 3.0-h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val #3

    return grad