"""
Dependencies:
tensorflow: 2.1.0
matplotlib
numpy
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
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
W1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
W2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
W3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

def forward(x):
    x = tf.reshape(x, [-1, 28*28])
    h1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)
    out = tf.matmul(h2, W3) + b3
    return out

# 定义损失函数
def loss_func(out, y):
    y_onehot = tf.one_hot(y, depth=10)
    loss = tf.reduce_mean(tf.square(y_onehot - out))
    return loss

train_ds, test_ds = prepare_dataset()

# 优化损失函数
learning_rate = 1e-3
def optimizer_func():
    for step, (x, y) in enumerate(train_ds):
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            out = forward(x)
            loss = loss_func(out, y)
        grads = tape.gradient(loss, [W1, b1, W2, b2, W3, b3])
        W1.assign_sub(learning_rate * grads[0])
        b1.assign_sub(learning_rate * grads[1])
        W2.assign_sub(learning_rate * grads[2])
        b2.assign_sub(learning_rate * grads[3])
        W3.assign_sub(learning_rate * grads[4])
        b3.assign_sub(learning_rate * grads[5])

# 测试预测效果
def evaluate_func():
    total_correct = 0
    total_sample = 0
    for x_test, y_test in test_ds:
        x_test = tf.reshape(x_test, [-1, 28 * 28])
        logits = forward(x_test)
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
    for epoch in range(100):
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
