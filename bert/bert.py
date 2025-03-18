import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import pandas as pd

# 定义数据集类
class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length):
        self.comments = comments
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        comment = str(self.comments[idx])
        label = self.labels[idx]
        # 调用分词器分词、编码、截断到max_length长度
        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            # attention_mask和input_ids长度相同的二进制向量
            # 对于原始文本标记，attention_mask对应位置取值为1，表示参与注意力计算
            # 对于填充的标记，attention_mask对应位置取值为0，表示不参与注意力计算
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载数据集
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    # 过滤标签为0，1
    df = df[df['label'].isin([0, 1])]
    comments = df['comment'].tolist()
    labels = df['label'].tolist()
    return comments, labels

# 训练模型
def train_model(model, train_dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}')

# 评估模型
def evaluate_model(model, test_dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            # logits维度为(batch_size, num_classes)
            predictions = torch.argmax(logits, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    print(f'Test Accuracy: {accuracy}')

# 进行预测
def predict(model, tokenizer, comment, device, max_length):
    model.eval()
    encoding = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # logits维度为(batch_size, num_classes)
        prediction = torch.argmax(logits, dim=1).item()

    return prediction

# 主函数
def main():
    # 配置参数
    train_file_path = 'train.csv'
    test_file_path = 'test.csv'
    # https://modelscope.cn/models/tiansz/bert-base-chinese
    model_name = 'tiansz/bert-base-chinese'
    max_length = 128
    batch_size = 16
    epochs = 10
    learning_rate = 2e-5

    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_comments, train_labels = load_dataset(train_file_path)
    test_comments, test_labels = load_dataset(test_file_path)

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    # 创建数据集和数据加载器
    train_dataset = CommentDataset(train_comments, train_labels, tokenizer, max_length)
    test_dataset = CommentDataset(test_comments, test_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_dataloader, optimizer, device, epochs)

    # 评估模型
    evaluate_model(model, test_dataloader, device)

    # 进行预测示例
    sample_comment = "货收到了！装上了！感觉高端霸气上档次"
    prediction = predict(model, tokenizer, sample_comment, device, max_length)
    sentiment = "正面" if prediction == 1 else "负面"
    print(f'评论: {sample_comment}')
    print(f'情感预测: {sentiment}')

if __name__ == "__main__":
    main()
