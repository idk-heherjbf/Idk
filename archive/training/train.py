import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sqlite3
import json
from model.architecture import TransformerModel
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

# Hyperparameters
BATC_SIZE = 32
SEQ_LEN = 64
EPOCHS = 10
LR = 1e-4

# SQLite setup
conn = sqlite3.connect("memory/training_trace.db")
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

# Dataset
class ShellDataset(Dataset):
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        self.samples = []
        for line in lines:
            trait = line.split(">")[0].replace("<", "")
            ids = tokenizer.encode(line).ids
            # Split long ids into multiple samples of length SEQ_LEN
            for i in range(0, len(ids) - 1, SEQ_LEN):
                chunk = ids[i:i+SEQ_LEN]
                if len(chunk) >= 2:
                    self.samples.append((chunk, trait))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, trait = self.samples[idx]
        x_ids = ids[:-1]
        y_ids = ids[1:]

        # Pad to SEQ_LEN - 1
        x_ids += [0] * (SEQ_LEN - 1 - len(x_ids))
        y_ids += [0] * (SEQ_LEN - 1 - len(y_ids))

        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)
        return x, y, trait

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training
dataset = ShellDataset("data/synthetic_shell_logs.txt")
loader = DataLoader(dataset, batch_size=BATC_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch_idx, (x, y, trait) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log to SQLite only every 100 batches
        if batch_idx % 100 == 0:
            cursor.execute("INSERT INTO trace (input, trait, loss) VALUES (?, ?, ?)",
                           (json.dumps(x[0].tolist()), trait[0], float(loss.item())))
            conn.commit()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

conn.close()
