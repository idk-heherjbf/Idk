import sys
import os
import hashlib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sqlite3
import json
from model.architecture import Seq2SeqTransformer
from tokenizer.simple_tokenizer import SimpleTokenizer

# Paths

DATA_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/cleaned_shell_logs.txt"))
MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model.pt"))
META_PATH: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/model_meta.json"))

# Ensure the trace table has the correct schema
memory_db_path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../memory/training_trace.db"))
conn = sqlite3.connect(memory_db_path)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch INTEGER,
    batch_idx INTEGER,
    loss REAL,
    accuracy REAL,
    timestamp DATETIME
)
""")
conn.commit()

# Initialize tokenizer (simpler approach)
special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
tokenizer = SimpleTokenizer(special_tokens=special_tokens)

# Build vocabulary from training data

# Overfit on a single example for debugging
single_question = "How do I scan all ports on a target with nmap?"
single_answer = "nmap -p- <target>"
texts = [single_question, single_answer]
tokenizer.build_vocab(texts)  # type: ignore
vocab_size = tokenizer.get_vocab_size()

# Reduced hyperparameters for better training
BATCH_SIZE = 4  # Smaller batch size
SRC_SEQ_LEN = 32  # Shorter sequences
TGT_SEQ_LEN = 32
EPOCHS = 50
LR = 5e-4  # Higher learning rate
MODEL_DIM = 256  # Smaller model
NHEAD = 8
NUM_LAYERS = 3  # Fewer layers

# SQLite setup
os.makedirs(os.path.dirname(memory_db_path), exist_ok=True)
conn = sqlite3.connect(memory_db_path)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS trace (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch INTEGER,
    batch_idx INTEGER,
    loss REAL,
    accuracy REAL,
    timestamp DATETIME
)
""")
conn.commit()


from typing import List, Tuple

class QADataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str):
        self.samples: List[Tuple[List[int], List[int]]] = []
        bos_id: int = tokenizer.token_to_id("<bos>")
        eos_id: int = tokenizer.token_to_id("<eos>")
        pad_id: int = tokenizer.token_to_id("<pad>")
        # Only one sample for overfit test
        question = single_question
        answer = single_answer
        q_ids: List[int] = tokenizer.encode(question)[:SRC_SEQ_LEN-1]
        a_ids: List[int] = tokenizer.encode(answer)[:TGT_SEQ_LEN-2]
        a_ids = [bos_id] + a_ids + [eos_id]
        q_ids += [pad_id] * (SRC_SEQ_LEN - len(q_ids))
        a_ids += [pad_id] * (TGT_SEQ_LEN - len(a_ids))
        self.samples.append((q_ids, a_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src, tgt = self.samples[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Vocabulary size: {vocab_size}")

model = Seq2SeqTransformer(
    vocab_size, 
    d_model=MODEL_DIM, 
    nhead=NHEAD, 
    num_encoder_layers=NUM_LAYERS,
    num_decoder_layers=NUM_LAYERS,
    dropout=0.1
).to(device)

# Initialize weights properly
def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        # m.bias is always a Parameter, so just set it
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, 0, 0.1)

model.apply(init_weights)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

pad_id = tokenizer.token_to_id("<pad>")
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

# Check for existing model

train_data_hash = hash_file(DATA_PATH)
model_exists = os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
should_train = True

# Only skip retraining if BOTH files exist and the hash matches
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
            should_train = True
    except Exception as e:
        print(f"Error loading model: {e}. Retraining.")
        should_train = True
else:
    print("Model or meta file missing. Retraining and saving model.")
    should_train = True

if should_train:

    dataset = QADataset(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("ERROR: No valid training samples found!")
        sys.exit(1)

    # For overfit test, use all data for training, skip validation
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    avg_train_loss = 0.0
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        model.train()
        total_train_loss = 0
        total_correct = 0
        total_tokens = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, vocab_size), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # type: ignore
            preds = torch.argmax(logits, dim=-1)
            mask = tgt_output != pad_id
            correct = ((preds == tgt_output) & mask).sum().item()
            tokens = mask.sum().item()
            total_train_loss += loss.item()
            total_correct += correct
            total_tokens += tokens
            if batch_idx % 10 == 0:
                accuracy = correct / max(1, tokens)
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}, Acc: {accuracy:.4f}")
                cursor.execute(
                    "INSERT INTO trace (epoch, batch_idx, loss, accuracy) VALUES (?, ?, ?, ?)",
                    (epoch + 1, batch_idx, loss.item(), accuracy)
                )
                conn.commit()
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        avg_train_acc = total_correct / max(1, total_tokens)
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
        # Save model every epoch
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        with open(META_PATH, "w") as f:
            json.dump({
                "train_data_hash": train_data_hash,
                "vocab_size": vocab_size,
                "model_dim": MODEL_DIM
            }, f)
        print(f"Model saved at epoch {epoch+1}.")
    print(f"\nTraining complete. Final train loss: {avg_train_loss:.4f}")

conn.close()