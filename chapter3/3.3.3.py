import numpy as np

X = np.array([1,2])
print("X={}, X.shape={}".format(X, X.shape))

W = np.array([[1,3,5], [2,4,6]])
print("W={}, W.shape={}".format(W, W.shape))

Y = np.dot(X, W)
print(Y)