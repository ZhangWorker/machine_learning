"""
Dependencies:
tensorflow: 2.1.0
matplotlib
numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# 获取训练样本
x = np.linspace(-1, 1, 100)
noise = np.random.normal(0.0, 0.01, size=x.shape)
y = 0.1 * x + 0.5 + noise


# plt.scatter(x, y)
# plt.show()

# 定义线性回顾权重和偏置
W = tf.Variable(tf.random.uniform([1], -1.0, 1.0), name='weight')
b = tf.Variable(tf.zeros([1]), name='biase')
y_hat = W * x + b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - y_hat), name='loss')
# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss, name='train')
# 创建会话
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for step in range(100):
    _, l, pred = sess.run([train_op, loss, y_hat])
    if step % 2 == 0:
        plt.cla()
        plt.scatter(x, y)
        plt.plot(x, pred, 'r-', lw=5)
        plt.pause(0.1)

plt.show()

