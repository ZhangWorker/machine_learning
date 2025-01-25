import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd
import jieba

# 调用jieba分词单个句子，分词后以空格分割字符
def cutting_words(text):
    # 精确模式分词
    seg_list = jieba.cut(text, cut_all=False)
    return " ".join(seg_list)

# 数据预处理类, 数据包含文本和标签
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len   # 每个句子最多保留50个字符

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 调用分词工具分词，以空格拼接字符
        cut_text = cutting_words(text)
        # 获取每个字符对应索引，获取不到就赋值为0
        text_indices = [self.vocab.get(word, 0) for word in cut_text.split()]
        # 句子长度小于50个字符，后面填充0
        if len(text_indices) < self.max_len:
            text_indices = text_indices + [0] * (self.max_len - len(text_indices))
        # 句子长度大于50个字符，截断50个字符
        else:
            text_indices = text_indices[:self.max_len]
        # 字符和标记转化为tensor
        return torch.tensor(text_indices), torch.tensor(label).long()

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        # 嵌入层, 将字符编码为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向多层LSTM层，包含2层LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            bidirectional=True, dropout=dropout, batch_first=True)
        # 全连接层，hidden_dim * 2表示拼接两个方向的隐藏状态
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # 嵌入层
        embedded = self.embedding(text)
        # LSTM层， hidden shape: (num_layers * num_directions, batch_size, hidden_size) -> (4, 32, 256)
        # cell表示记忆单元, cell shape: (num_layers * num_directions, batch_size, hidden_size) -> (4, 32, 256)
        output, (hidden, cell) = self.lstm(embedded)
        # 拼接双向LSTM的隐藏状态
        # hidden[-2, :, :] 最后一层正向隐藏状态，hidden[-1, :, :] 最后一层反向隐藏状态
        # hidden[-4, :, :] 倒数第二层正向隐藏状态，hidden[-3, :, :] 倒数第二层反向隐藏状态
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # 全连接层, hidden shape: (batch_size, 2 * hidden_size) -> (32, 512)
        return self.fc(hidden.squeeze(0))

# 构建词汇表, 包含字符和对应的索引
def build_vocab(texts, min_freq=5):
    counter = Counter()
    for text in texts:
        # 调用分词工具分词，以空格拼接字符
        text = cutting_words(text)
        # 记录字符和对应的频率
        counter.update(text.split())
    # 过滤低频词
    counter = {word: freq for (word, freq) in counter.items() if freq >= min_freq}
    # 过滤后的字符构建字符和索引的映射
    vocab = {word: idx + 1 for idx, (word, freq) in enumerate(counter.items())}
    # 未知字符索引为默认值0
    vocab['<unk>'] = 0
    return vocab

# 数据预处理
def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    # 二分类，标签取值0或者1
    df = df[df['label'].isin([0, 1])]
    texts = df['text'].values
    labels = df['label'].values
    # 划分训练集和测试集，80%样本训练，20%样本测试
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
    # 构建词汇表
    vocab = build_vocab(train_texts)
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)
    return train_dataset, test_dataset, vocab

# 训练模型
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()    # 设置为训练模式
    total_loss = 0
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 评估模型
def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # 设置为测试模式
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # 不参与梯度计算
        for texts, labels in test_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)   # 预测概率最大的标签
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return total_loss / len(test_loader), accuracy

# 主函数
def main():
    data_path = "./train.csv"
    train_dataset, test_dataset, vocab = preprocess_data(data_path)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 2  # 二分类任务
    num_layers = 2
    dropout = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化模型
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    # 定义交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练10个epochs
    num_epochs = 10
    for epoch in range(num_epochs):
         train_loss = train_model(model, train_loader, criterion, optimizer, device)
         test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
         print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    main()