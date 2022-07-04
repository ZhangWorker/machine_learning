"""
Dependencies:
tensorflow: 2.1.0
matplotlib
numpy
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 准备数据集
batch_size = 128
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def prepare_dataset():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(preprocess).shuffle(100000).batch(batch_size=batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess).batch(batch_size=batch_size)
    return train_ds, test_ds

# 定义网络结构
def define_network():
    conv_network = Sequential([
        layers.Conv2D(32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2),
        layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu),
        layers.MaxPool2D(pool_size=[2, 2], strides=2)
    ])

    full_network = Sequential([
        layers.Dense(512, activation=tf.nn.relu),
        layers.Dense(128, activation=tf.nn.relu),
        layers.Dense(10, activation=None)
    ])

    conv_network.build(input_shape=[None, 28, 28, 1])
    full_network.build(input_shape=[None, 64 * 7 * 7])
    return conv_network, full_network

# 定义损失函数
def loss_func(logits, y):
    y_onehot = tf.one_hot(y, depth=10)
    loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True))
    return loss

optimizer = optimizers.Adam(lr = 1e-3)
conv_network, full_network = define_network()
train_ds, test_ds = prepare_dataset()
variables = conv_network.trainable_variables + full_network.trainable_variables

# 优化损失函数
def optimizer_func():
    for step, (x, y) in enumerate(train_ds):
        x = tf.expand_dims(x, -1)
        with tf.GradientTape() as tape:
            out = conv_network(x)
            out = tf.reshape(out, [-1, 64 * 7 * 7])
            logits = full_network(out)
            loss = loss_func(logits, y)
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

# 测试预测效果
def evaluate_func():
    total_correct = 0
    total_sample = 0
    for x_test, y_test in test_ds:
        x_test = tf.expand_dims(x_test, -1)
        out = conv_network(x_test)
        out = tf.reshape(out, [-1, 64 * 7 * 7])
        logits = full_network(out)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
        correct = tf.equal(pred, y_test)
        correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))
        total_correct += int(correct)
        total_sample += x_test.shape[0]
    accuracy = total_correct / total_sample * 100
    return accuracy


epoches = []
accuracies = []
def main():
    for epoch in range(10):
        optimizer_func()
        accuracy = evaluate_func()
        epoches.append(epoch)
        accuracies.append(accuracy)
    plt.plot(epoches, accuracies, 'r-', lw=2)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()

if __name__ == '__main__':
    main()