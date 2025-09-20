from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
import os

# Paths
DATA_PATH = "data/synthetic_shell_logs.txt"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

# Normalizer: lowercase, strip accents
normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# Pre-tokenizer: whitespace
pre_tokenizer = pre_tokenizers.Whitespace()

# Model: BPE
model = models.BPE()

# Trainer: trait-aware vocab
trainer = trainers.BpeTrainer(
    vocab_size=8192,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<fuzzing>", "<teardown>", "<stealth>", "<chain>", "<default>"]
)

# Tokenizer
tokenizer = Tokenizer(model)
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.decoder = decoders.BPEDecoder()
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
)

# Train
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing training data: {DATA_PATH}")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]
tokenizer.train_from_iterator(lines, trainer)

# Save
tokenizer.save(TOKENIZER_PATH)
print(f"Tokenizer saved to {TOKENIZER_PATH}")
