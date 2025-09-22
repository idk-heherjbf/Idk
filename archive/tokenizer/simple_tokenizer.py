import re
from collections import defaultdict

class SimpleTokenizer:
    def __init__(self, special_tokens=None):
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab = {i: tok for tok, i in self.vocab.items()}
        self.next_id = len(self.vocab)

    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.inv_vocab[self.next_id] = token
                    self.next_id += 1

    def tokenize(self, text):
        # Simple whitespace + punctuation split
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        return " ".join(tokens)

    def token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    def id_to_token(self, idx):
        return self.inv_vocab.get(idx, "<unk>")

    def get_vocab_size(self):
        return len(self.vocab)
