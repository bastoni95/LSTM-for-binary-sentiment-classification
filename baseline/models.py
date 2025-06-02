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

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        # 词嵌入层：将词ID转换为对应的词向量，padding
        # _idx=0用于忽略填充词
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多层感知机：一层隐藏层 + ReLU激活 + 输出层
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, **kwargs):
        emb = self.embed(x)               # 输入x的词向量表示，形状[B, L, D]
        avg = emb.mean(dim=1)             # 对序列长度维度求平均，得到句子向量[B, D]
        return self.net(avg)              # 经过MLP得到分类结果


class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=(3,4,5), num_filters=100):
        super().__init__()
        # 词嵌入层
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多个不同大小卷积核的卷积层，卷积核宽度等于词向量维度
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)   # Dropout防止过拟合
        # 全连接层，将拼接后的卷积输出映射到类别数
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, **kwargs):
        emb = self.embed(x).unsqueeze(1)                # [B, 1, L, D]，添加通道维度，方便卷积操作
        convs = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]  # 卷积+激活，去掉宽度维度，形状[B, C, L']
        pools = [torch.max(c, dim=2)[0] for c in convs] # 对每个卷积输出做最大池化，得到[B, C]
        concat = torch.cat(pools, dim=1)                # 拼接所有池化结果，形状[B, C * 卷积核数]
        out = self.dropout(concat)                       # Dropout正则化
        return self.fc(out)                              # 输出分类结果

