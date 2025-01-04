import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import ssl
ssl._create_default_https_context = ssl._create_unverified_context



# 定义数据预处理的转换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量, 将ndarray数组转换为Tensor数据类型
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化，MNIST数据集的均值和标准差进行数据的标准化，即减去均值除以方差，此时均值0.1307和方差0.3081是MNIST数据集计算好的数据，直接使用即可
])

# 加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义网络结构
class DnnNet(nn.Module):
    def __init__(self):
        super(DnnNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # 输入层到隐藏层，MNIST图像展平后是784(28*28)维
        self.relu = nn.ReLU()           # relu激活函数，做非线性变化
        self.fc2 = nn.Linear(128, 10)   # 隐藏层到输出层，对应10个数字类别

    def forward(self, x):
        x = x.view(-1, 784)  # 展平输入图像张量 (batch_size, 28, 28) -> (batch_size, 28*28)
        x = self.fc1(x)      # shape: (batch_size, 784) -> (batch_size, 128)
        x = self.relu(x)     # shape: (batch_size, 128) -> (batch_size, 128)
        x = self.fc2(x)      # shape: (batch_size, 128) -> (batch_size, 10)
        return x

model = DnnNet()

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，适合多分类任务
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器，可设置学习率和动量等参数

epochs = 10  # 训练轮数，可以根据实际情况调整
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()   # 梯度清零
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()        # 反向传播计算梯度 dy / dw
        optimizer.step()       # 更新参数    w = w - lr * dy / dw

        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / 100}')
            running_loss = 0.0

correct = 0
total = 0
with torch.no_grad():  # 不参与反向计算梯度
    model.eval()
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')