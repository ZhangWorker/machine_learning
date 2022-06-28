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
n_data = np.ones((100, 2))
x0 = np.random.normal(2*n_data, 1)
y0 = np.zeros(100)
x1 = np.random.normal(-2*n_data, 1)
y1 = np.ones(100)
x = np.vstack((x0, x1))
y = np.hstack((y0, y1))


# plt.scatter(x[:, 0], x[:, 1], c=y, s=100, lw=0, cmap='RdYlGn')
# plt.show()

# 定义逻辑回顾权重和偏置
W = tf.Variable(tf.random.uniform([2, 2], -1.0, 1.0), name='weight')
b = tf.Variable(tf.zeros([2]), name='biase')
# 获取预测输出
tf_x = tf.placeholder(tf.float32, x.shape)
tf_y = tf.placeholder(tf.int32, y.shape)
output = tf.matmul(tf_x, W) + b
# 定义损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
# 梯度下降优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train_op = optimizer.minimize(loss, name='train')
# 训练精度
accuracy = tf.metrics.accuracy(
    labels=tf.squeeze(tf_y), predictions=tf.argmax(output, axis=1))[1]
# 创建会话
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# 分类边界 w1*x1+w2*x2+b=0
def x2(x1, W, b):
    return (-b[1] - W[1, 0] * x1) / W[1, 1]

x1 = np.linspace(-4, 4, 100)

plt.ion()
for step in range(100):
    _, acc, pred, weight, bias = sess.run([train_op, accuracy, output, W, b], {tf_x: x, tf_y: y})
    if step % 2 == 0:
        plt.cla()
        plt.scatter(x[:, 0], x[:, 1], c=pred.argmax(1), s=100, lw=0, cmap='RdYlGn')
        plt.plot(x1, x2(x1, weight, bias), 'b-', lw=5)
        plt.text(1.5, -4, 'Accuracy=%.2f' % acc, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()




