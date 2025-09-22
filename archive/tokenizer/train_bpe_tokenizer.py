from tokenizers import Tokenizer, models, trainers

# Read pentesting Q&A data
import os
data_path = os.path.join(os.path.dirname(__file__), '../data/cleaned_shell_logs.txt')
with open(os.path.abspath(data_path), 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    corpus = []
    for line in lines:
        if '?' in line and '?' != line[0]:
            q, a = line.split('?', 1)
            corpus.append(q.strip() + '?')
            corpus.append(a.strip())

tokenizer_model = models.BPE()
bpe_tokenizer = Tokenizer(tokenizer_model)
trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])
bpe_tokenizer.train_from_iterator(corpus, trainer)
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../tokenizer/tokenizer.json'))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
bpe_tokenizer.save(save_path)
print('New BPE tokenizer trained and saved.')
