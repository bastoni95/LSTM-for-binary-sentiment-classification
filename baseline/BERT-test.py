import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 自定义Dataset，用于加载CSV数据并做tokenizer编码
class BERTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        df = pd.read_csv(file_path)
        self.encodings = tokenizer(df['sentence'].tolist(), truncation=True, padding=True, max_length=max_len)
        self.labels = df['label'].tolist()
        self.sentences = df['sentence'].tolist()
    def __getitem__(self, idx):
        # 返回单条数据，包含输入ids、attention_mask等，以及标签和原文本
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['text'] = self.sentences[idx]
        return item
    def __len__(self):
        return len(self.labels)

# 测试函数：对测试集预测并收集结果
def test_model(model, loader, device):
    model.eval()
    all_preds, all_labels, all_texts = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            texts = batch.pop("text")                 # 取出原文本
            batch = {k: v.to(device) for k, v in batch.items()}  # 移动数据到device
            outputs = model(**batch)                   # 前向传播
            preds = torch.argmax(outputs.logits, dim=1)  # 取最大概率类别
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_texts.extend(texts)
    return all_texts, all_labels, all_preds

# 设备选择，GPU优先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练tokenizer和模型，并加载训练好的权重
tokenizer = BertTokenizer.from_pretrained("E:\\科研训练\\LLM-Model\\bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("E:\\科研训练\\LLM-Model\\bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load("model/best_model_bert.pt", map_location=device))
model.to(device)

# 准备测试集DataLoader
test_dataset = BERTDataset("data/test.csv", tokenizer)
test_loader = DataLoader(test_dataset, batch_size=16)

# 运行测试并收集结果
texts, labels, preds = test_model(model, test_loader, device)

# 保存预测结果到CSV
df_result = pd.DataFrame({
    "sentence": texts,
    "label": labels,
    "predicted": preds
})
df_result.to_csv("bert_test_predictions.csv", index=False, encoding="utf-8-sig")

# 计算四个常用分类指标
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, average="binary")
recall = recall_score(labels, preds, average="binary")
f1 = f1_score(labels, preds, average="binary")

# 打印指标
print(f"\n 测试集评估指标：")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
