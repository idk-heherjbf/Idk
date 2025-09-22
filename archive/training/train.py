
import sys
import os
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.functional import log_softmax
import sqlite3
import json
from model.architecture import Seq2SeqTransformer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



# Use HuggingFace BertWordPieceTokenizer
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cleaned_shell_logs.txt"))
from pathlib import Path
tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenizer/hf_wordpiece"))
os.makedirs(tokenizer_path, exist_ok=True)
special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
texts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "?" in line and "?" != line[0]:
            parts = line.split("?", 1)
            question = parts[0].strip() + "?"
            answer = parts[1].strip()
            if answer.startswith("A ") or answer.startswith("a "):
                answer = answer[2:].strip()
            if answer:
                texts.append(question)
                texts.append(answer)
corpus_path = os.path.join(tokenizer_path, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")
from tokenizers import Tokenizer
import os
tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tokenizer/tokenizer.json'))
print(f"Loading tokenizer from: {tokenizer_path}")
tokenizer = Tokenizer.from_file(tokenizer_path)
# Add special tokens if not present
for tok in special_tokens:
    if tokenizer.token_to_id(tok) is None:
        tokenizer.add_special_tokens([tok])
vocab_size = tokenizer.get_vocab_size()


# Model save/load paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model.pt"))
META_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model_meta.json"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cleaned_shell_logs.txt"))


# Hyperparameters
BATC_SIZE = 8
SRC_SEQ_LEN = 48  # question length
TGT_SEQ_LEN = 48  # answer length
EPOCHS = 30
LR = 1e-4
MODEL_DIM = 512
NHEAD = 8



# SQLite setup (use absolute path)
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "../memory/training_trace.db")
MEMORY_DB_PATH = os.path.abspath(MEMORY_DB_PATH)
conn = sqlite3.connect(MEMORY_DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    input TEXT,
    trait TEXT,
    loss REAL
)
""")
conn.commit()

class QADataset(Dataset):
    def __init__(self, path):
        self.samples = []
        bos_id = tokenizer.token_to_id("[BOS]")
        eos_id = tokenizer.token_to_id("[EOS]")
        pad_id = tokenizer.token_to_id("[PAD]")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "?" in line and "?" != line[0]:
                    parts = line.split("?", 1)
                    question = parts[0].strip() + "?"
                    answer = parts[1].strip()
                    if answer.startswith("A ") or answer.startswith("a "):
                        answer = answer[2:].strip()
                    if answer:
                        q_ids = tokenizer.encode(question).ids[:SRC_SEQ_LEN]
                        a_ids = [bos_id] + tokenizer.encode(answer).ids[:TGT_SEQ_LEN-2] + [eos_id]
                        q_ids += [pad_id] * (SRC_SEQ_LEN - len(q_ids))
                        a_ids += [pad_id] * (TGT_SEQ_LEN - len(a_ids))
                        self.samples.append((q_ids, a_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# Utility: hash file
def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()




# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqTransformer(vocab_size, d_model=MODEL_DIM, nhead=NHEAD).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

def lr_lambda(step):
    warmup_steps = 4000
    d_model = MODEL_DIM
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

scheduler = LambdaLR(optimizer, lr_lambda)
pad_id = tokenizer.token_to_id("<pad>")
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=None):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.ignore_index = ignore_index
    def forward(self, x, target):
        x = x.log_softmax(dim=-1)
        true_dist = torch.zeros_like(x)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        ignore = (target == self.ignore_index)
        target = target.masked_fill(ignore, 0)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist.masked_fill_(ignore.unsqueeze(1), 0)
        return torch.mean(torch.sum(-true_dist * x, dim=-1))

criterion = LabelSmoothingLoss(vocab_size, smoothing=0.1, ignore_index=pad_id)

# Check for existing model and metadata

import os
import json
train_data_hash = hash_file(DATA_PATH)
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
should_train = True

# Force retrain if vocab size changes (model/tokenizer update)
if model_exists:
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
        if meta.get("train_data_hash") == train_data_hash:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Loaded trained model from disk. Skipping retraining.")
            should_train = False
        else:
            print("Training data changed. Retraining model.")
    except Exception as e:
        print(f"Model/tokenizer mismatch or error: {e}\nDeleting old model and retraining.")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        if os.path.exists(META_PATH):
            os.remove(META_PATH)
        should_train = True


if should_train:
    from torch.utils.data import random_split
    dataset = QADataset(DATA_PATH)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATC_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATC_SIZE)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 3

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    smoothing = SmoothingFunction().method1
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch+1}/{EPOCHS}")
        model.train()
        loss = None
        total_acc = 0
        total_bleu = 0
        total_samples = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(src, tgt_input)
                    loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(src, tgt_input)
                loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            # Accuracy: compare predicted tokens to tgt_output (ignoring pad)
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_output != pad_id
            correct = (preds == tgt_output) & mask
            acc = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
            total_acc += acc * src.size(0)
            # BLEU: compute for first sample in batch (for speed)
            pred_seq = preds[0].tolist()
            tgt_seq = tgt_output[0].tolist()
            pred_seq = [t for t in pred_seq if t != pad_id]
            tgt_seq = [t for t in tgt_seq if t != pad_id]
            # Convert token IDs to strings for BLEU
            pred_str = [str(t) for t in pred_seq]
            tgt_str = [str(t) for t in tgt_seq]
            bleu_score = 0.0
            if len(pred_str) > 0 and len(tgt_str) > 0:
                bleu_score = sentence_bleu([tgt_str], pred_str, smoothing_function=smoothing)
            if isinstance(bleu_score, float):
                total_bleu += bleu_score
            else:
                total_bleu += 0.0
            total_samples += src.size(0)
            if batch_idx % 100 == 0:
                cursor.execute("INSERT INTO trace (input, trait, loss) VALUES (?, ?, ?)",
                               (json.dumps(src[0].tolist()), "Q&A", float(loss.item())))
                conn.commit()
                print(f"Epoch {epoch+1}/{EPOCHS} Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} Acc: {acc:.4f} BLEU: {total_bleu/(batch_idx+1):.4f}")
        if loss is not None:
            print(f"Epoch {epoch+1}/{EPOCHS} complete. Last batch loss: {loss.item():.4f} Avg Acc: {total_acc/total_samples:.4f} Avg BLEU: {total_bleu/len(train_loader):.4f}")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS} complete. No batches processed.")

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_acc = 0
        val_bleu = 0
        val_samples = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                logits = model(src, tgt_input)
                vloss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
                val_loss += vloss.item()
                val_batches += 1
                preds = torch.argmax(logits, dim=-1)
                mask = tgt_output != pad_id
                correct = (preds == tgt_output) & mask
                acc = correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0
                val_acc += acc * src.size(0)
                pred_seq = preds[0].tolist()
                tgt_seq = tgt_output[0].tolist()
                pred_seq = [t for t in pred_seq if t != pad_id]
                tgt_seq = [t for t in tgt_seq if t != pad_id]
                # Convert token IDs to strings for BLEU
                pred_str = [str(t) for t in pred_seq]
                tgt_str = [str(t) for t in tgt_seq]
                bleu_score = 0.0
                if len(pred_str) > 0 and len(tgt_str) > 0:
                    bleu_score = sentence_bleu([tgt_str], pred_str, smoothing_function=smoothing)
                if isinstance(bleu_score, float):
                    val_bleu += bleu_score
                else:
                    val_bleu += 0.0
                val_samples += src.size(0)
        val_loss /= max(1, val_batches)
        avg_val_acc = val_acc / max(1, val_samples)
        avg_val_bleu = val_bleu / max(1, val_batches)
        print(f"Validation loss after epoch {epoch+1}: {val_loss:.4f} Avg Acc: {avg_val_acc:.4f} Avg BLEU: {avg_val_bleu:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model so far
            torch.save(model.state_dict(), MODEL_PATH)
            with open(META_PATH, "w") as f:
                json.dump({"train_data_hash": train_data_hash}, f)
            print(f"New best model saved to {MODEL_PATH}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    # Load best model after training
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")

conn.close()
