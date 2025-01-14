import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(256),             # 调整图像大小为256*256
    transforms.CenterCrop(224),         # 从中心裁剪出224*224的图像
    transforms.ToTensor(),              # 将图像转化为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 对图像进行归一化处理，包含三个通道的均值和方差，即减去均值除以方差
])

# 加载训练集
train_dataset = datasets.ImageNet(root='./data', split='train', download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 加载测试集
test_dataset = datasets.ImageNet(root='./data', split='val', download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# 定义VGG16模型
class VGG16(nn.Module):
   def __init__(self, num_classes=1000):
       super(VGG16, self).__init__()
       self.features = nn.Sequential(
           nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),    # 图片shape: (128, 3, 224, 224) -> (128, 64, 224, 224)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果,
           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),   # 图片shape: (128, 64, 224, 224) -> (128, 64, 224, 224)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.MaxPool2d(kernel_size=2, stride=2),                                  # 图片shape: (128, 64, 224, 224) -> (128, 64, 112, 112)
           nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # 图片shape: (128, 64, 112, 112) -> (128, 128, 112, 112)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # 图片shape: (128, 128, 112, 112) -> (128, 128, 112, 112)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.MaxPool2d(kernel_size=2, stride=2),                                  # 图片shape: (128, 128, 112, 112) -> (128, 128, 56, 56)
           nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # 图片shape: (128, 128, 56, 56) -> (128, 256, 56, 56)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # 图片shape: (128, 256, 56, 56) -> (128, 256, 56, 56)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), # 图片shape: (128, 256, 56, 56) -> (128, 256, 56, 56)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.MaxPool2d(kernel_size=2, stride=2),                                  # 图片shape: (128, 256, 56, 56) -> (128, 256, 28, 28)
           nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 256, 28, 28) -> (128, 512, 28, 28)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 512, 28, 28) -> (128, 512, 28, 28)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 512, 28, 28) -> (128, 512, 28, 28)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.MaxPool2d(kernel_size=2, stride=2),                                  # 图片shape: (128, 512, 28, 28) -> (128, 512, 14, 14)
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 512, 14, 14) -> (128, 512, 14, 14)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 512, 14, 14) -> (128, 512, 14, 14)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1), # 图片shape: (128, 512, 14, 14) -> (128, 512, 14, 14)
           nn.ReLU(inplace=True),                                                  # 激活函数，inplace=True直接修改输入张量的结果
           nn.MaxPool2d(kernel_size=2, stride=2)                                   # 图片shape: (128, 512, 14, 14) -> (128, 512, 7, 7)
       )
       self.classifier = nn.Sequential(
           nn.Linear(512 * 7 * 7, 4096),    # 图片shape: (128, 512 * 7 * 7) -> (128, 4096)
           nn.ReLU(inplace=True),           # 激活函数，inplace=True直接修改输入张量的结果
           nn.Dropout(),                    # dropout，随机删除全连接网络的部分连接
           nn.Linear(4096, 4096),           # 图片shape: (128, 4096) -> (128, 4096)
           nn.ReLU(inplace=True),           # 激活函数，inplace=True直接修改输入张量的结果
           nn.Dropout(),                    # dropout，随机删除全连接网络的部分连接
           nn.Linear(4096, num_classes)     # 图片shape: (128, 4096) -> (128, 1000)
       )

   def forward(self, x):
       x = self.features(x)                 # 图片shape: (128, 3, 224, 224) -> (128, 512, 7, 7)
       x = x.view(x.size(0), -1)            # 图片shape: (128, 512, 7, 7) -> (128, 512 * 7 * 7)
       x = self.classifier(x)               # 图片shape: (128, 512 * 7 * 7) -> (128, 1000)
       return x

model = VGG16()

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