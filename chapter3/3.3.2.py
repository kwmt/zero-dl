import numpy as np

A = np.array([[1,2], [3,4]])
print("A={}".format(A))

B = np.array([[5,6], [7,8]])
print("B={}".format(B))

dot = np.dot(A,B)
print("dot={}".format(dot))


A = np.array([[1,2,3], [4,5,6]])
print("A={}".format(A))

B = np.array([[1,2], [3,4], [5,6]])
print("B={}".format(B))


dot = np.dot(A,B)
print("dot={}".format(dot))


A = np.array([[1,2], [3,4], [5,6]])
print("A={}".format(A))
print("A.shape={}".format(A.shape))

B = np.array([7,8])
print("B={}".format(B))

dot = np.dot(A,B) 
print("dot={}".format(dot))
