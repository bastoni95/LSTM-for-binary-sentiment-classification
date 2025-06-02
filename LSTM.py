import os               
import re              

import torch            
import torch.nn as nn   
import torch.optim as optim  
from torch.utils.data import Dataset, DataLoader  

import pandas as pd     
import numpy as np      
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.decomposition import PCA  
from sklearn.manifold import TSNE     
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 下载nltk资源
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# 初始化停用词和词形还原工具
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ------------------------- 数据预处理函数 -------------------------
def tokenize(text, remove_stopwords=True, lemmatize=True):
    """
    1. 确保输入为字符串
    2. 转为小写
    3. 使用正则去除标点符号，仅保留字母、数字和空格
    4. 可选地去除停用词
    5. 可选地词形还原
    6. 按空格拆分为 token 列表
    """
    text = str(text).lower()  # 小写化
    text = re.sub(r"[^a-z0-9\s]", "", text)  # 去除标点符号
    tokens = word_tokenize(text)  # 使用 word_tokenize 进行分词

    # 可选：去除停用词
    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

    # 可选：词形还原
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def build_vocab(sent_list, min_freq=2):
    """
    基于 token 列表构建词典：
    - 统计每个 token 的出现频率
    - 过滤低于 min_freq 的 token
    - 特殊符号 <pad>=0, <unk>=1
    返回: {token: index}
    """
    freq = {}
    # 统计所有句子中 token 频率
    for tokens in sent_list:
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    # 初始化词典，保留 pad 和 unk
    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    # 按出现频率筛选 token
    for tok, count in freq.items():
        if count >= min_freq:
            vocab[tok] = idx
            idx += 1
    return vocab

def encode(tokens, vocab):
    """
    将分词列表转换为索引列表：
    - vocab 中不存在的 token 标记为 <unk>
    """
    return [vocab.get(tok, vocab['<unk>']) for tok in tokens]


def collate_fn(batch):
    """
    DataLoader 批处理函数：
    - 输入 batch: List[(token_idx_list, label)]
    - 对 token 列表进行 padding，统一到 batch 中最长序列长度
    - 返回 LongTensor 格式的 inputs 和 labels
    """
    sentences, labels = zip(*batch)
    # 找出 batch 中最大长度
    max_len = max(len(s) for s in sentences)
    # 对所有句子进行 pad，使用 0（<pad>）
    padded = [s + [0] * (max_len - len(s)) for s in sentences]
    inputs = torch.tensor(padded, dtype=torch.long)
    targets = torch.tensor(labels, dtype=torch.long)
    return inputs, targets

# ------------------------- 自定义 Dataset -------------------------
class SST2Dataset(Dataset):
    """
    继承 torch.utils.data.Dataset：
    - 读取 GLUE SST-2 格式的 train.tsv/dev.tsv
    - text 列为句子，label 列为二分类标签
    - 如果未传入 vocab，则在训练时构建词典；验证时可复用该词典
    """
    def __init__(self, file_path, vocab=None, min_freq=2):
        # 读取 tsv
        df = pd.read_csv(file_path, sep=',') 
    
        # 对所有句子进行分词
        self.sentences = [tokenize(sent) for sent in df['sentence'].tolist()]
        self.labels = df['label'].tolist()
        # 构建或复用词典
        if vocab is None:
            self.vocab = build_vocab(self.sentences, min_freq)
        else:
            self.vocab = vocab
        # 将分词结果转换为索引序列
        self.data = [encode(tokens, self.vocab) for tokens in self.sentences]

    def __len__(self):
        # 返回数据集大小
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回第 idx 条样本的 (input_idxs, label)
        return self.data[idx], self.labels[idx]
    
# ------------------------- LSTM 类 -------------------------
class ManualLSTM(nn.Module):
    """
    手动实现的单层 LSTM 文本分类器，支持多种池化策略：
    - pooling: 'last'(最后一步), 'mean'(平均池化), 'max'(最大池化)
    - 支持 pad 掩码(mask) 用于屏蔽填充位置
    - 可选是否返回中间隐藏层表示（用于可视化）
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0,
                 dropout: float = 0.5,
                 pooling: str = 'mean'):
        super().__init__()

        # 词嵌入层，padding_idx 保证 PAD 位置始终为 0 向量
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        inp_size = embed_dim + hidden_dim  # 输入为 x_t 和 h_{t-1} 的拼接

        # 四个门的线性映射层
        self.linear_f = nn.Linear(inp_size, hidden_dim)  # 遗忘门
        self.linear_i = nn.Linear(inp_size, hidden_dim)  # 输入门
        self.linear_c = nn.Linear(inp_size, hidden_dim)  # 候选记忆
        self.linear_o = nn.Linear(inp_size, hidden_dim)  # 输出门

        # 分类头：将最后的表示映射到类别空间
        self.fc = nn.Linear(hidden_dim, num_classes)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化所有线性层的参数"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self,
                x: torch.LongTensor,                 # 输入：[B, L]
                mask: torch.BoolTensor = None,       # 掩码：[B, L]
                init_states: tuple = None,           # 初始状态 (h0, c0)
                return_hidden: bool = False          # 是否返回所有隐藏状态
                ) -> tuple:
        """
        前向传播函数。
        返回：
        - logits: [B, num_classes]
        - rep: 最终表示 [B, H]
        - all_hidden: [B, L, H]（可选，用于可视化）
        """
        batch, seq_len = x.size()

        # 词嵌入 + dropout
        emb = self.dropout(self.embed(x))  # [B, L, D]

        # 若未提供 mask，则默认全部为有效位置
        if mask is None:
            mask = torch.ones(batch, seq_len, dtype=torch.bool, device=x.device)

        # 初始化 h 和 c 为 0
        if init_states is None:
            h = emb.new_zeros(batch, self.hidden_dim)
            c = emb.new_zeros(batch, self.hidden_dim)
        else:
            h, c = init_states

        all_hidden = []  # 存储每个时间步的隐藏状态

        # 时间步展开
        for t in range(seq_len):
            x_t = emb[:, t, :]                     # 当前输入 [B, D]
            combined = torch.cat([x_t, h], dim=1)  # 拼接 h_{t-1} 和 x_t

            # LSTM 四个门
            f = torch.sigmoid(self.linear_f(combined))        # 遗忘门
            i = torch.sigmoid(self.linear_i(combined))        # 输入门
            c_tilde = torch.tanh(self.linear_c(combined))     # 候选记忆
            c = f * c + i * c_tilde                            # 新记忆状态
            o = torch.sigmoid(self.linear_o(combined))        # 输出门
            h_new = o * torch.tanh(c)                          # 新隐藏状态

            # 如果当前位置是 PAD，则保持上一个状态
            m = mask[:, t].unsqueeze(1).float()
            h = h_new * m + h * (1 - m)
            c = c * m + c * (1 - m)

            # dropout 可选
            h = self.dropout(h)

            all_hidden.append(h)  # 存储当前时间步隐藏状态

        # 合并为 [B, L, H]
        all_hidden = torch.stack(all_hidden, dim=1)

        # 池化策略选择（将 L 个向量合并为 1 个表示 rep）
        if self.pooling == 'mean':
            rep = (all_hidden * mask.unsqueeze(-1)).sum(dim=1) / \
                  mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        elif self.pooling == 'max':
            rep = all_hidden.masked_fill(~mask.unsqueeze(-1), float('-inf')).max(dim=1).values
        else:  # 'last'
            last_indices = mask.sum(dim=1).long() - 1  # 找到每个样本的最后有效位置
            rep = all_hidden[torch.arange(batch, device=x.device), last_indices]

        # 分类预测
        logits = self.fc(rep)

        # 返回值配置
        if return_hidden:
            return logits, rep, all_hidden  # 提供中间层用于可视化
        else:
            return logits, rep
        
# ======================= 训练函数 =======================
def train_one_epoch(model, dataloader, optimizer, criterion, device, vocab):
    """
    执行模型的一个训练轮次，带 tqdm 进度条。
    """
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    # tqdm 包装 dataloader，显示训练进度
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        mask = inputs != vocab['<pad>']

        optimizer.zero_grad()
        outputs, _ = model(inputs, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# ======================= 验证函数 =======================
def evaluate(model, dataloader, criterion, device, vocab):
    """
    模型评估函数（验证集/测试集均可），带 tqdm 进度条。
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            mask = inputs != vocab['<pad>']
            outputs, _ = model(inputs, mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# ======================= 测试 + 可视化函数 =======================
def test_and_visualize(model, dataloader, criterion, device, vocab, perplexity=30):
    """
    模型测试与可视化函数：带 tqdm 进度条，输出准确率，精确率，召回率，F1分数，并绘制 PCA & t-SNE。
    """
    model.eval()
    all_preds, all_labels, all_reps = [], [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            mask = inputs != vocab['<pad>']
            outputs, reps, _ = model(inputs, mask, return_hidden=True)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_reps.append(reps.cpu())

    # 拼接所有 batch 的表示
    all_reps_tensor = torch.cat(all_reps, dim=0)
    all_labels_tensor = torch.tensor(all_labels)

    # 计算并打印准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n Test Accuracy: {accuracy:.4f}")

    # 计算精确率，召回率，F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 计算并显示混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(len(set(all_labels))), yticklabels=range(len(set(all_labels))))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

    # 可视化
    plot_pca(all_reps_tensor, all_labels_tensor, title="Test Set - PCA")
    plot_tsne(all_reps_tensor, all_labels_tensor, title="Test Set - t-SNE", perplexity=perplexity)

# ======================= 可视化函数 =======================
def plot_pca(hidden_states: torch.Tensor, labels: torch.Tensor, title: str = "PCA Visualization"):
    hidden_np = hidden_states.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(hidden_np)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_tsne(hidden_states: torch.Tensor, labels: torch.Tensor, title: str = "t-SNE Visualization", perplexity: int = 30):
    hidden_np = hidden_states.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    reduced = tsne.fit_transform(hidden_np)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels_np, cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(scatter, label='Class Label')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show() 