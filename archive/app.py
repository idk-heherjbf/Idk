from fastapi import Body
# ...existing code...

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import torch
import json
from archive.model.architecture import Seq2SeqTransformer
from archive.tokenizer.simple_tokenizer import SimpleTokenizer

# ...existing code...
app = FastAPI()
# ...existing code...


# Nucleus sampling for seq2seq Q&A with parameter replacement
import re
import numpy as np
def extract_target(text):
    # Extract IP or domain from user input (simple regex)
    ip_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", text)
    if ip_match:
        return ip_match.group(1)
    domain_match = re.search(r"([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
    if domain_match:
        return domain_match.group(1)
    return None

def nucleus_sample(probs, p=0.9):
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumulative_probs, p)
    filtered_indices = sorted_indices[:cutoff+1]
    filtered_probs = probs[filtered_indices]
    filtered_probs /= filtered_probs.sum()
    return np.random.choice(filtered_indices, p=filtered_probs)

def generate_answer(question, max_len=64, nucleus_p=0.9):
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not loaded. Please train first.")
    model.eval()
    # Parameter replacement: extract <target> from user input
    target = extract_target(question)
    q = question
    if target:
        q = re.sub(r"<target>", target, question)
    q_ids = tokenizer.encode(q)[:63]
    q_ids += [tokenizer.token_to_id("<pad>")] * (64 - len(q_ids))
    src = torch.tensor([q_ids], dtype=torch.long).to(device)
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    tgt = torch.tensor([[bos_id]], dtype=torch.long).to(device)
    output_ids = []
    for _ in range(max_len):
        with torch.no_grad():
            logits = model(src, tgt)
            logits = logits[0, -1].cpu().numpy()
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            next_token = nucleus_sample(probs, p=nucleus_p)
        if next_token == eos_id or next_token == pad_id:
            break
        output_ids.append(next_token)
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
    print(f"[DIAG] Input: {question}")
    print(f"[DIAG] Tokenized input: {tokenizer.tokenize(question)}")
    print(f"[DIAG] Output token IDs: {output_ids}")
    print(f"[DIAG] Decoded output: {tokenizer.decode(output_ids)}")
    return tokenizer.decode(output_ids)

@app.post("/api/ask")
async def api_ask(data: dict = Body(...)):
    user_input = data.get("input", "")
    if not model or not tokenizer:
        return JSONResponse({"error": "Model not loaded. Please train first."}, status_code=400)
    try:
        output = generate_answer(user_input)
        return JSONResponse({"input": user_input, "output": output})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve static files (for frontend)
import pathlib
BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

MODEL_PATH = BASE_DIR / "model" / "model.pt"
META_PATH = BASE_DIR / "model" / "model_meta.json"
TOKENIZER_PATH = BASE_DIR / "tokenizer" / "tokenizer.json"

tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = None

def load_model():
    global model, tokenizer, vocab_size
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    DATA_PATH = BASE_DIR / "data" / "cleaned_shell_logs.txt"
    tokenizer = SimpleTokenizer(special_tokens=special_tokens)
    # Build vocab from data
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
    tokenizer.build_vocab(texts)
    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqTransformer(vocab_size).to(device)
    if os.path.exists(str(MODEL_PATH)):
        model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
        model.eval()
        return True
    return False

# Load model at startup if available
load_model()

@app.get("/", response_class=HTMLResponse)
def index():
    index_path = STATIC_DIR / "index.html"
    with open(index_path) as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

@app.get("/train")
def train():
    # Run the training script
    train_script = BASE_DIR / "training" / "train.py"
    def stream():
        process = subprocess.Popen(["python3", str(train_script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if process.stdout is not None:
            for line in process.stdout:
                yield f"data: {line}\n\n"
        process.wait()
        load_model()
        yield f"data: Training complete. Model reloaded.\n\n"
    from fastapi.responses import StreamingResponse
    return StreamingResponse(stream(), media_type="text/event-stream")

@app.post("/test")
async def test(request: Request):
    data = await request.json()
    user_input = data.get("input", "")
    if not model or not tokenizer:
        return JSONResponse({"error": "Model not loaded. Please train first."}, status_code=400)
    output = generate_answer(user_input)
    return JSONResponse({"input": user_input, "output": output})

@app.post("/reload")
def reload_model():
    loaded = load_model()
    if loaded:
        return JSONResponse({"result": "Model reloaded from disk."})
    else:
        return JSONResponse({"error": "Model file not found. Please train first."}, status_code=400)
