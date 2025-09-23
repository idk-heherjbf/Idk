import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()  # type: ignore
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)].to(x.device)


# Custom Seq2Seq Transformer for Q&A
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 4, num_decoder_layers: int = 4, dropout: float = 0.1):
        super().__init__()  # type: ignore
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer.encoder(src_emb)
        return memory

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        output = self.transformer.decoder(tgt_emb, memory)
        return self.fc_out(output)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src)
        output = self.decode(tgt, memory)
        return output
