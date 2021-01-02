import numpy as np
import softmax

a = np.array([0.3, 2.9 ,4.0])
y = softmax.softmax(a)
print(y)

print(np.sum(y))