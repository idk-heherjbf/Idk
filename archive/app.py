from fastapi import Body
# ...existing code...

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import subprocess
import os
import torch
from archive.model.architecture import Seq2SeqTransformer
from archive.tokenizer.simple_tokenizer import SimpleTokenizer

# ...existing code...
app = FastAPI()
# ...existing code...


# Nucleus sampling for seq2seq Q&A with parameter replacement
import re
import numpy as np
def extract_target(text: str) -> str:
    # Extract IP or domain from user input (simple regex)
    ip_match = re.search(r"(\d{1,3}(?:\.\d{1,3}){3})", text)
    if ip_match:
        return ip_match.group(1)
    domain_match = re.search(r"([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", text)
    if domain_match:
        return domain_match.group(1)
    return ""

import numpy as np

def nucleus_sample(probs: np.ndarray, p: float = 0.9) -> int:
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff: int = int(np.searchsorted(cumulative_probs, p))
    filtered_indices = sorted_indices[:cutoff+1]
    filtered_probs = probs[filtered_indices]
    filtered_probs /= filtered_probs.sum()
    return np.random.choice(filtered_indices, p=filtered_probs)

def generate_answer(question: str, max_len: int = 64, nucleus_p: float = 0.9) -> str:
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
    last_token = None
    repeat_count = 0
    for step in range(max_len):
        with torch.no_grad():
            logits = model(src, tgt)
            logits = logits[0, -1].cpu().numpy()
            # Greedy decoding for debugging
            next_token = int(np.argmax(logits))
            # Optionally, use nucleus_sample for more diversity:
            # probs = np.exp(logits - np.max(logits))
            # probs = probs / probs.sum()
            # next_token = nucleus_sample(probs, p=nucleus_p)
        print(f"[GEN] Step {step}: token_id={next_token}, token='{tokenizer.id_to_token(int(next_token))}'")
        if next_token == eos_id or next_token == pad_id or next_token == tokenizer.token_to_id('<unk>'):
            print("[GEN] Stopping: <eos>, <pad>, or <unk> token generated.")
            break
        if last_token is not None and next_token == last_token:
            repeat_count += 1
        else:
            repeat_count = 0
        if repeat_count >= 2:
            print("[GEN] Stopping: token repeated 2 times.")
            break
        output_ids.append(next_token)  # type: ignore
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        last_token = next_token
    print(f"[DIAG] Input: {question}")
    print(f"[DIAG] Tokenized input: {tokenizer.tokenize(question)}")
    print(f"[DIAG] Output token IDs: {output_ids}")
    print(f"[DIAG] Decoded output: {tokenizer.decode(output_ids)}")  # type: ignore
    return tokenizer.decode(output_ids)  # type: ignore

# Training improvement suggestion:
# - Try EPOCHS=100, LR=1e-4, BATCH_SIZE=8 for better convergence.

@app.post("/api/ask")
async def api_ask(data: dict[str, str] = Body(...)):
    user_input = data.get("input", "")
    if not model or not tokenizer:
        return JSONResponse({"error": "Model not loaded. Please train first."}, status_code=400)
    try:
        output = generate_answer(user_input)
        return JSONResponse({"input": user_input, "output": output})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Serve static files (for frontend)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "model/model.pt"))
META_PATH = os.path.abspath(os.path.join(BASE_DIR, "model/model_meta.json"))
TOKENIZER_PATH = os.path.abspath(os.path.join(BASE_DIR, "tokenizer/tokenizer.json"))

tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = None

def load_model():
    global model, tokenizer, vocab_size
    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "data/cleaned_shell_logs.txt"))
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
                    texts.append(question)  # type: ignore
                    texts.append(answer)  # type: ignore
    tokenizer.build_vocab(texts)  # type: ignore
    vocab_size = tokenizer.get_vocab_size()
    model = Seq2SeqTransformer(
        vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1
    ).to(device)
    if os.path.exists(str(MODEL_PATH)):
        try:
            model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
            model.eval()
            return True
        except Exception as e:
            print(f"Model/tokenizer mismatch or error: {e}\nDeleting old model and retraining.")
            if os.path.exists(str(MODEL_PATH)):
                os.remove(str(MODEL_PATH))
            if os.path.exists(str(META_PATH)):
                os.remove(str(META_PATH))
            # Optionally, retrain here or return False to trigger retrain via API
            return False
    return False

# Load model at startup if available
load_model()

@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path) as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)

@app.get("/train")
def train():
    # Run the training script
    train_script = os.path.join(BASE_DIR, "training", "train.py")
    def stream():
        process = subprocess.Popen(["python3", train_script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
