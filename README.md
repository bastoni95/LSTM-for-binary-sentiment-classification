# 项目名称：LSTM for binary sentiment classification  
> A minimal single-layer LSTM with custom word embeddings for binary sentiment classification and feature visualization.

---

## 项目简介  
本项目基于 PyTorch 从零手写实现了一个单层 LSTM，专为 SST-2 二分类情感分析任务设计。  
区别于常见的封装模型（如 `nn.LSTM`），本项目完整复现了 LSTM 的门控机制、状态更新、序列建模流程，并结合自定义词嵌入与可视化模块，构建一个便于教学、实验和研究的端到端文本分类系统。

---

## 模型亮点  
- 手写实现 LSTM 单元核心逻辑（完全不依赖 `nn.LSTM`）  
- 精确控制时间步状态转移与门控计算（`f/i/o/c` 四门）  
- 支持多种池化策略：`mean` / `max` / `last`  
- 使用自定义词向量嵌入，不依赖预训练模型  
- 支持 `padding mask`，避免填充信息污染学习过程  
- 提供完整中间层特征可视化（混淆矩阵、PCA、t-SNE）  

---

## 项目结构  
```

├── LSTM.py         # 模型结构 + 数据预处理 + 可视化函数（核心逻辑）
├── train.py        # 训练脚本（包括 early stopping、保存模型、绘图等）
├── test.py         # 测试脚本（加载模型并进行评估和特征可视化）
├── vocab.json      # 训练阶段生成的词表映射（token → index）
├── data/
│   ├── train.csv   # 训练集（必须包含 sentence 和 label 两列）
│   ├── dev.csv     # 验证集
│   └── test.csv    # 测试集
├── baseline/
│   ├── models.py           # MLP 与 CNN 模型结构定义
│   ├── run-baselines.py    # MLP / CNN 的训练脚本
│   ├── test-baseline.py    # MLP / CNN 的测试评估脚本
│   ├── BERT-train.py       # BERT 模型训练脚本
│   └── BERT-test.py        # BERT 模型测试脚本

````

---

## 快速开始

### 1. 训练模型

运行以下命令启动训练：

```bash
python train.py
````

训练过程中将执行以下操作：

* 自动保存模型参数到 model/ 目录：

  * `best_model.pt`：验证集准确率最佳的模型
  * `last_model.pt`：最后一个 epoch 的模型
* 自动生成词表文件 `vocab.json`（仅首次训练）
* 输出训练和验证的 loss / accuracy，并绘制学习曲线图
* 自动绘制训练曲线图（如 train\_val\_curve.png，路径可在代码中自定义）

模型默认配置：

* 嵌入维度：100
* 隐藏层维度：128
* Dropout：0.5
* 池化策略：mean
* 优化器：Adam
* 学习率：1e-3
* 批大小：32
* Early stopping：验证集准确率连续 3 次无提升将终止训练

---

### 2. 数据准备说明

将数据文件放入 `data/` 目录，格式如下：

```csv
sentence,label
"this movie is great",1
"the plot is boring",0
```

确保以下三个文件均存在：

* `train.csv`：训练集
* `dev.csv`：验证集
* `test.csv`：测试集

---

### 3. 测试模型并进行可视化

运行以下命令进行测试与可视化分析：

```bash
python test.py
```

测试过程中将会：

* 自动加载 `model/best_model.pt`（可在代码中修改为 `last_model.pt`）
* 使用测试集进行推理并评估以下指标：

  * Accuracy
  * Precision / Recall / F1 Score
  * 混淆矩阵（图示输出）
* 使用 `test_and_visualize()` 函数降维可视化中间层表示：

  * PCA 投影
  * t-SNE 聚类（默认 perplexity = 30，可修改）

---

## 模型对比基线（Baseline Models）

为全面评估手写 LSTM 的性能，项目还集成了以下三种典型的文本分类基线模型，可在相同数据集上进行训练与测试对比：

### 支持的模型类型

| 模型名称 | 特点简介                                   |
| ---- | -------------------------------------- |
| MLP  | 基于平均词向量的多层感知机，结构简单，训练迅速                |
| CNN  | TextCNN 架构，提取局部 n-gram 特征，捕捉局部上下文      |
| BERT | 基于 `bert-base-uncased` 预训练语言模型，语义建模能力强 |

### Baseline 训练命令

运行以下命令，将依次训练 MLP 和 CNN 模型：

```bash
python run-baselines.py
```

训练结束后将在 `model/` 目录生成以下模型参数文件：

```
model/
├── best_model_MLP.pt
├── last_model_MLP.pt
├── best_model_CNN.pt
└── last_model_CNN.pt
```

### Baseline 测试与评估

运行以下命令测试 MLP 和 CNN 模型在测试集上的表现：

```bash
python test-baseline.py
```

输出将包括每个模型的：

* Accuracy
* Precision / Recall / F1 Score
* 平均损失（loss）

### BERT 模型训练与测试

训练 BERT：

```bash
python BERT-train.py
```

训练过程中：

* 使用 bert-base-uncased 预训练模型作为基础，进行微调（finetune）
* 使用训练集和验证集进行迭代优化，支持早停机制
* 自动保存训练过程中最优模型权重到 model/best\_model\_bert.pt
* 生成并保存训练过程的曲线图（如 bert\_training\_curve.png），用于分析模型收敛情况

测试 BERT：

```bash
python BERT-test.py
```

测试完成后将：

* 自动加载 model/best\_model\_bert.pt 最佳模型权重
* 在测试集上计算准确率（Accuracy）、精确率（Precision）、召回率（Recall）和 F1 分数，并打印至控制台
* 生成预测结果文件 bert\_test\_predictions.csv，包含句子、真实标签和预测标签，方便进一步分析

测试结束后将在当前目录下输出：

* bert\_test\_predictions.csv：预测结果
* bert\_training\_curve.png：训练曲线图
* 控制台显示 Accuracy、Precision、Recall、F1 等评估指标

---

## 可视化样例

测试完成后，你将获得以下可视化图示：

* 混淆矩阵（Confusion Matrix）
* PCA 主成分分析二维投影（PCA Projection）
* t-SNE 聚类可视化（t-SNE Visualization）

你可以在 `LSTM.py` 中自定义：

* 降维方式（`plot_pca` / `plot_tsne`）
* 配色方案
* 图像尺寸与风格

---

## 数据格式要求

确保数据文件为 `.csv` 格式，必须包含如下字段：

```csv
sentence,label
"this movie is great",1
"the plot is boring",0
```

* `sentence`：输入文本（英文）
* `label`：标签，支持二分类（0 或 1）

---

## 致谢

本项目为高校机器学习课程的课设项目，旨在通过从零实现 LSTM 网络，帮助初学者深入理解循环神经网络的内部机制及其在情感分析任务中的实际应用。
在此特别感谢以下支持与资源：

* PyTorch 官方文档与社区：提供了深度学习模型实现的基础框架与指导。
* 斯坦福大学 Sentiment Treebank (SST-2) 数据集：为情感分类提供了标准评测数据。
* HuggingFace Transformers 库：用于实现与对比 BERT 模型，拓展了模型对比的深度。
* 教师教学与课程资源：为项目设计、模型实现和结果分析提供了有力支持。

此外，感谢各类开源项目、博客、论文和 LLM 模型对本项目设计思路与实现方法的启发与借鉴。

本项目面向初学者，旨在帮助大家深入理解神经网络在情感分类上的应用，欢迎提出建议与改进，共同完善！

