import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 导入自定义的 Dataset、模型和 collate_fn
from LSTM import SST2Dataset, ManualLSTM, collate_fn, train_one_epoch, evaluate

# —— 参数配置 ——#
device = torch.device('cuda')
batch_size = 32           # 每个 batch 样本数
num_epochs = 10           # 总训练轮数
lr = 1e-3                 # 初始学习率
num_classes = 2           # 分类数
patience = 3              # 早停机制的耐心参数


# 训练集和验证集的文件路径
train_path = 'data/train.csv'  # 训练集路径
dev_path = 'data/dev.csv'     # 验证集路径

# 构建 Dataset（第一次读取时自动建立 vocab）
train_ds = SST2Dataset(file_path=train_path)
dev_ds = SST2Dataset(file_path=dev_path, vocab=train_ds.vocab)  

# print(f"训练集样本数: {len(train_ds)}")
# print(f"验证集样本数: {len(dev_ds)}")
# print("训练集前几个样本:", train_ds[:5])  

# 1. 将 vocab 保存到 vocab.json
with open('vocab.json', 'w', encoding='utf-8') as f:
    json.dump(train_ds.vocab, f, ensure_ascii=False, indent=2)
print(" vocab 已保存到 vocab.json")

# DataLoader 会按 batch 迭代，并自动调用 collate_fn 做 padding
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
print(f"训练数据集的样本数: {len(train_loader.dataset)}")
print(f"验证数据集的样本数: {len(dev_loader.dataset)}")

# —— 模型 & 损失 & 优化器 ——#
model = ManualLSTM(
    vocab_size=len(train_ds.vocab),      # 词表大小
    embed_dim=100,                       # 词嵌入维度
    hidden_dim=128,                      # LSTM 隐藏层维度
    num_classes=num_classes,             # 输出类别数
    pad_idx=train_ds.vocab['<pad>'],     # PAD 索引
    dropout=0.5,                         # Dropout 比例
    pooling='mean'                       # 池化方式（mean/last/max）
).to(device)  # 将模型加载到 GPU

criterion = nn.CrossEntropyLoss()            # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam 优化器

# —— 训练循环 ——#
best_val_acc = 0.0  # 用于追踪最佳验证准确率
early_stopping_counter = 0  # 早停计数器
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(1, num_epochs + 1):

    # 1) 训练一个 epoch
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, device, train_ds.vocab
    )

    # 2) 在验证集上评估
    val_loss, val_acc = evaluate(
        model, dev_loader, criterion, device, train_ds.vocab
    )

    # 3) 打印本轮结果
    print(f"[Epoch {epoch}/{num_epochs}] "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
          f"Dev: loss={val_loss:.4f}, acc={val_acc:.4f}")

    # 4) 保存训练和验证的统计数据
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # 5) 如果验证准确率提升，则保存当前模型参数
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("model", exist_ok=True) 
        torch.save(model.state_dict(), "model/best_model.pt")
        print(f" Best model saved at epoch {epoch}, dev acc={val_acc:.4f}")
        early_stopping_counter = 0  # Reset early stopping counter
    else:
        early_stopping_counter += 1

    # 6) 早停：如果连续 patience 次验证准确率没有提升，则停止训练
    if early_stopping_counter >= patience:
        print(f" Early stopping at epoch {epoch}")
        break

# —— 训练结束后保存最后一轮模型 ——#
os.makedirs("model", exist_ok=True) 
torch.save(model.state_dict(), "model/last_model.pt")
print(" Training complete. Final model saved to last_model.pt")

# —— 可视化训练过程 ——#
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
