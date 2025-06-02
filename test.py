import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LSTM import SST2Dataset, ManualLSTM, collate_fn, test_and_visualize

# —— 参数配置 ——#
device      = torch.device('cuda')
batch_size  = 32    # 测试时也用同样的 batch size
num_classes = 2     # 与训练时保持一致


# 1. 从 vocab.json 中加载词表
with open('vocab.json', 'r', encoding='utf-8') as f:
    train_vocab = json.load(f)

# 2. 构建测试集，复用已保存的 vocab
test_path   = 'data/test.csv'  #
test_ds   = SST2Dataset(test_path, vocab=train_vocab)
# 3. DataLoader 会按 batch 迭代，并自动调用 collate_fn 做 padding
test_loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# —— 模型初始化 & 加载参数 ——#
model = ManualLSTM(
    vocab_size  = len(test_ds.vocab),
    embed_dim   = 100,
    hidden_dim  = 128,
    num_classes = num_classes,
    pad_idx     = test_ds.vocab['<pad>'],
    dropout     = 0.5,
    pooling     = 'mean'
).to(device)

# 加载你希望使用的模型文件（best_model.pt 或 last_model.pt）
checkpoint = torch.load("model/best_model.pt", map_location=device, weights_only=True)   
model.load_state_dict(checkpoint)
print(" Loaded model parameters from best_model.pt")

criterion = nn.CrossEntropyLoss()  # 损失函数无需在测试时使用，但保持接口一致

# —— 测试 & 可视化 ——#
test_and_visualize(
    model, test_loader, criterion, device, test_ds.vocab,
    perplexity=30
)
