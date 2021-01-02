import numpy as np

A = np.array([1,2,3,4])

print(A)
ndim = np.ndim(A)
print(ndim)
print(A.shape)
print(A.shape[0])

B = np.array([[1,2], [3,4]])
print("B={}".format(B))
print("np.ndim(B)= {}".format( np.ndim(B)))
print("B.shape={}".format(B.shape))