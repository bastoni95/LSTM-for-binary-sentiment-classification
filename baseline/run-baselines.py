import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MLPClassifier, CNNTextClassifier, SST2Dataset, collate_fn
from tqdm import tqdm

# --------- 训练一个epoch的标准训练函数 ---------
def train_one_epoch_standard(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    loop = tqdm(loader, desc="Training", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()            # 梯度清零
        out = model(x)                  # 前向传播
        loss = criterion(out, y)         # 计算损失
        loss.backward()                 # 反向传播
        optimizer.step()                # 参数更新

        total_loss += loss.item()
        preds = out.argmax(dim=1)       # 预测类别
        correct += (preds == y).sum().item()
        total += y.size(0)
        loop.set_postfix(loss=loss.item(), acc=correct / total)  # 显示当前loss和准确率
    return total_loss / len(loader), correct / total


# --------- 验证/测试的评估函数 ---------
def evaluate_standard(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            loop.set_postfix(acc=correct / total)  # 显示当前准确率
    return total_loss / len(loader), correct / total


# --------- 配置参数 ---------
device = torch.device("cuda")   # 使用GPU
batch_size = 32
num_epochs = 20
lr = 1e-3
patience = 3                   # 早停等待轮数
embed_dim = 100
hidden_dim = 128
num_classes = 2

# --------- 模型选择函数 ---------
def get_model(name, vocab_size, pad_idx):
    if name == 'MLP':
        return MLPClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
    elif name == 'CNN':
        return CNNTextClassifier(vocab_size, embed_dim, num_classes)
    else:
        raise ValueError(f"Unsupported model name: {name}")

# --------- 数据集准备 ---------
train_path = 'data/train.csv'
dev_path = 'data/dev.csv'
test_path = 'data/test.csv'

train_ds = SST2Dataset(train_path)               # 训练集
dev_ds = SST2Dataset(dev_path, vocab=train_ds.vocab)  # 验证集，共用训练集词表
test_ds = SST2Dataset(test_path, vocab=train_ds.vocab) # 测试集，共用词表

# 保存词表
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(train_ds.vocab, f, indent=2)

pad_idx = train_ds.vocab["<pad>"]

# 构建DataLoader
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# --------- 主训练流程 ---------
for model_name in ["MLP", "CNN"]:
    print(f"\n Training model: {model_name}")
    model = get_model(model_name, len(train_ds.vocab), pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch_standard(model, train_loader, optimizer, criterion, device)
        # 验证模型性能
        val_loss, val_acc = evaluate_standard(model, dev_loader, criterion, device)

        print(f"[{model_name}] Epoch {epoch} | Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # 保存最优模型，重置早停计数
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), f"model/best_model_{model_name}.pt")
            print(f" Best model saved for {model_name} at epoch {epoch}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 达到早停条件则退出训练
        if early_stop_counter >= patience:
            print(f" Early stopping for {model_name}")
            break

    # 保存最后一次训练的模型权重
    torch.save(model.state_dict(), f"model/last_model_{model_name}.pt")
