import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt

# 自定义Dataset，加载CSV文件并用BERT tokenizer编码文本
class BERTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        # 读取CSV，期望包含 'sentence' 和 'label' 两列
        df = pd.read_csv(file_path)
        # 使用tokenizer对句子进行编码，自动截断或填充至max_len长度
        self.encodings = tokenizer(df['sentence'].tolist(),
                                   truncation=True,
                                   padding=True,
                                   max_length=max_len)
        self.labels = df['label'].tolist()

    def __getitem__(self, idx):
        # 返回单条数据，包括input_ids, attention_mask等编码信息和对应标签
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # 返回数据集大小
        return len(self.labels)

# 验证函数，计算给定数据加载器上的准确率
def evaluate_bert(model, loader, device):
    model.eval()  # 进入评估模式，关闭dropout等
    correct, total = 0, 0
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():  # 禁止梯度计算，节省内存和计算
        for batch in loop:
            # 将batch中的数据转到GPU或CPU
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)  # 模型前向推理
            preds = outputs.logits.argmax(dim=1)  # 取预测最大概率类别
            correct += (preds == batch["labels"]).sum().item()  # 累计预测正确数
            total += batch["labels"].size(0)  # 累计样本数
            acc = correct / total  # 当前累计准确率
            loop.set_postfix(acc=acc)  # 进度条显示准确率
    return acc

# 设备选择，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据路径和训练超参
train_path = "data/train.csv"
dev_path = "data/dev.csv"
lr = 2e-5
batch_size = 16
num_epochs = 5
patience = 2  # 早停阈值：连续多少轮验证准确率不提升则停止训练

# 加载预训练BERT tokenizer和分类模型（本地路径）
tokenizer = BertTokenizer.from_pretrained("E:\\科研训练\\LLM-Model\\bert-base-uncased")
bert_model = BertForSequenceClassification.from_pretrained("E:\\科研训练\\LLM-Model\\bert-base-uncased",num_labels=2).to(device)

# 构建训练和验证数据集及其DataLoader
train_bert = BERTDataset(train_path, tokenizer)
val_bert = BERTDataset(dev_path, tokenizer)
train_loader = DataLoader(train_bert, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_bert, batch_size=batch_size)

# 优化器，AdamW适合Transformer微调
optimizer = AdamW(bert_model.parameters(), lr=lr)

# 记录训练过程的loss和验证准确率
train_losses, val_accuracies = [], []
best_val_acc = 0.0  # 记录最高验证准确率
early_stopping_counter = 0  # 连续无提升轮数计数器

# 训练循环，遍历每个epoch
for epoch in range(1, num_epochs + 1):
    bert_model.train()  # 训练模式，启用dropout等
    total_loss = 0
    loop = tqdm(train_loader, desc=f"[BERT] Epoch {epoch}")

    for batch in loop:
        # 送入设备
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = bert_model(**batch)  # 前向计算，outputs包含loss和logits
        loss = outputs.loss

        loss.backward()   # 反向传播，计算梯度
        optimizer.step()  # 优化器更新参数
        optimizer.zero_grad()  # 梯度清零，准备下一步

        total_loss += loss.item()  # 累计loss
        loop.set_postfix(loss=loss.item())  # 显示当前batch loss

    avg_train_loss = total_loss / len(train_loader)  # 计算平均训练loss

    # 验证集评估
    val_acc = evaluate_bert(bert_model, val_loader, device)

    train_losses.append(avg_train_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # 保存最优模型并重置早停计数
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(bert_model.state_dict(), "model/best_model_bert.pt")
        print(f" Best BERT model saved at epoch {epoch}")
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1  # 验证准确率无提升计数

    # 达到早停条件时退出训练
    if early_stopping_counter >= patience:
        print(f" Early stopping at epoch {epoch}")
        break

# 保存训练结束时的模型参数
torch.save(bert_model.state_dict(), "model/last_model_bert.pt")

# 绘制训练曲线（loss和准确率）
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Val Accuracy")
plt.title("BERT Training Curve")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("bert_training_curve.png")
plt.close()

print(" BERT training complete.")
