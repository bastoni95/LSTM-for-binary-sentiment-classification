import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from LSTM import SST2Dataset, collate_fn
from models import MLPClassifier, CNNTextClassifier


def evaluate_standard(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels)
            loop.set_postfix(acc=acc)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, prec, rec, f1

# ---------------- 配置 ---------------- #
device = torch.device("cuda")
model_names = ["MLP", "CNN"]
batch_size = 32

# ---------------- 加载词表和数据 ---------------- #
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

pad_idx = vocab.get("<pad>", 0)
test_ds = SST2Dataset("data/test.csv", vocab=vocab)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------- 模型工厂 ---------------- #
def get_model(name, vocab_size):
    if name == 'MLP':
        return MLPClassifier(vocab_size, 100, 128, 2)
    elif name == 'CNN':
        return CNNTextClassifier(vocab_size, 100, 2)
    else:
        raise ValueError("Unknown model name")

# ---------------- 逐模型评估 ---------------- #
criterion = torch.nn.CrossEntropyLoss()
results = []

for name in model_names:
    print(f"\n📦 Testing model: {name}")
    model = get_model(name, len(vocab)).to(device)
    model.load_state_dict(torch.load(f"model/best_model_{name}.pt", map_location=device, weights_only=True))
    model.eval()

    val_loss, acc, prec, rec, f1 = evaluate_standard(model, test_loader, criterion, device)
    results.append((name, acc, prec, rec, f1, val_loss))
    print(f"✅ {name} Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, Loss: {val_loss:.4f}")

# ---------------- 汇总结果 ---------------- #
print("\n📊 Baseline Test Results")
print("| Model   | Acc   | Prec  | Rec   | F1    | Loss  |")
print("|---------|-------|-------|-------|-------|--------|")
for name, acc, prec, rec, f1, loss in results:
    print(f"| {name:<7} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {loss:.4f} |")
