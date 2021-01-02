import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.layers import *
from common.util import im2col, col2im
import numpy as np


class SimpleConvNet:
    def __init__(self. input_dim=(1, 28, 28),
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100,
                output_size=10,
                weight_init_std=0.01):
        
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        # 畳み込み層の出力サイズ計算
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2)

        # 重みパラメータ初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.rand(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 必要なレイヤを生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad)
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grad['W1'] = self.layers['Conv1'].dW
        grad['b1'] = self.layers['Conv1'].db
        grad['W2'] = self.layers['Affine1'].dW
        grad['b2'] = self.layers['Affine1'].db
        grad['W3'] = self.layers['Affine2'].dW
        grad['b3'] = self.layers['Affine2'].db

        return grads



