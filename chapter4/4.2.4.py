import numpy as np


# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# # [2]を正解とする
# t = [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
# y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

# print(cross_entropy_error( np.array(y), np.array(t)))

# バッチ

# (x_train, t_train), (x_test, t_test) = \
#     load_mnist(normalize=True, one_hot_label=True)

# print(x_train.shape)
# print(t_train.shape)

# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# print(batch_mask)
# print(x_batch)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    print(batch_size)
    return -np.sum(np.log(y[np.arrange(batch_size), t])) / batch_size

t = [  0,    0,   1,   0,    0,   0,   0,   0,   0,   0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

print(cross_entropy_error( np.array(y), np.array(t)))