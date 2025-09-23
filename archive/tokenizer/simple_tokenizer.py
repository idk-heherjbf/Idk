import re


from typing import List, Dict, Optional

class SimpleTokenizer:
    def __init__(self, special_tokens: Optional[List[str]] = None) -> None:
        self.special_tokens: List[str] = special_tokens or ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.vocab: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        self.inv_vocab: Dict[int, str] = {i: tok for tok, i in self.vocab.items()}
        self.next_id: int = len(self.vocab)

    def build_vocab(self, texts: List[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.vocab:
                    self.vocab[token] = self.next_id
                    self.inv_vocab[self.next_id] = token
                    self.next_id += 1

    def tokenize(self, text: str) -> List[str]:
        # Simple whitespace + punctuation split
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def encode(self, text: str) -> List[int]:
        tokens: List[str] = self.tokenize(text)
        ids: List[int] = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens: List[str] = [self.inv_vocab.get(i, "<unk>") for i in ids]
        return " ".join(tokens)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def id_to_token(self, idx: int) -> str:
        return self.inv_vocab.get(idx, "<unk>")

    def get_vocab_size(self) -> int:
        return len(self.vocab)
