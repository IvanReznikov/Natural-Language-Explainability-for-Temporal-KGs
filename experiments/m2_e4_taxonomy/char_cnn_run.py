from pathlib import Path
import json
from datetime import datetime
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

SEED = 13
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "experiments" / "m2_e4" / "data" / "splits"
TASK_PREFIX = "result_taxonomy"

LABEL_KEYS = ["label", "result_type", "type"]
TEXT_KEYS = ["text", "result", "narrative", "summary", "content"]


def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def example_to_xy(rec):
    label = None
    for lk in LABEL_KEYS:
        if lk in rec:
            label = rec[lk]
            break
    texts = []
    for tk in TEXT_KEYS:
        if tk in rec and rec[tk]:
            texts.append(str(rec[tk]))
    if not texts:
        texts = [str(v) for k, v in rec.items() if k not in LABEL_KEYS]
    if label is None:
        raise ValueError(f"No label key in {rec.keys()}")
    return "\n".join(texts), label


def load_split(name):
    path = DATA_DIR / f"{TASK_PREFIX}_{name}.jsonl"
    return [example_to_xy(r) for r in load_jsonl(path)]


train = load_split("train")
val = load_split("val")
test = load_split("test")

labels = sorted({y for _, y in train})
label_to_id = {l: i for i, l in enumerate(labels)}

chars = [chr(i) for i in range(32, 127)]
char_to_id = {c: i + 1 for i, c in enumerate(chars)}  # 0 pad
PAD = 0


def encode(text, max_len=400):
    text = text.lower()
    ids = [char_to_id.get(ch, 0) for ch in text][:max_len]
    if len(ids) < max_len:
        ids += [PAD] * (max_len - len(ids))
    return ids


class CharDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(encode(x), dtype=torch.long), label_to_id[y]


train_ds = CharDataset(train)
val_ds = CharDataset(val)
test_ds = CharDataset(test)

BATCH = 64
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)
test_loader = DataLoader(test_ds, batch_size=BATCH)


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=32,
        num_filters=64,
        kernel_sizes=(3, 4, 5),
        dropout=0.3,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embed(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))
            h = torch.max(h, dim=2).values
            pooled.append(h)
        h_cat = torch.cat(pooled, dim=1)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


model = CharCNN(vocab_size=len(char_to_id) + 1, num_classes=len(labels)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)


def run_epoch(loader, train_mode=False):
    if train_mode:
        model.train()
    else:
        model.eval()
    tot_loss = 0.0
    tot = 0
    tot_correct = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train_mode:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(xb)
                loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        tot_correct += (preds == yb).sum().item()
        tot_loss += loss.item() * xb.size(0)
        tot += xb.size(0)
    return tot_loss / max(1, tot), tot_correct / max(1, tot)


best_state = None
best_val = 0.0
history = []
EPOCHS = 12
PATIENCE = 3
pat_left = PATIENCE
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train_mode=True)
    va_loss, va_acc = run_epoch(val_loader, train_mode=False)
    history.append(
        {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
        }
    )
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.3f} train_acc={tr_acc:.3f} | val_loss={va_loss:.3f} val_acc={va_acc:.3f}"
    )
    if va_acc > best_val:
        best_val = va_acc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        pat_left = PATIENCE
    else:
        pat_left -= 1
        if pat_left <= 0:
            print("Early stop")
            break

if best_state:
    model.load_state_dict(best_state)


def evaluate(loader):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            ys.extend(yb.cpu().tolist())
            ps.extend(pred.cpu().tolist())
    acc = accuracy_score(ys, ps)
    macro = f1_score(ys, ps, average="macro")
    return acc, macro


val_acc, val_f1 = evaluate(val_loader)
test_acc, test_f1 = evaluate(test_loader)
print(f"Val acc={val_acc:.3f} macro_f1={val_f1:.3f}")
print(f"Test acc={test_acc:.3f} macro_f1={test_f1:.3f}")

out_dir = ROOT / "output" / "m2_e4_taxonomy" / "e4a_taxonomy"
out_dir.mkdir(parents=True, exist_ok=True)
metrics_path = out_dir / "char_cnn.json"
with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(
        {
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "test_acc": test_acc,
            "test_macro_f1": test_f1,
            "history": history,
        },
        f,
        indent=2,
    )
print("Saved metrics to", metrics_path)
