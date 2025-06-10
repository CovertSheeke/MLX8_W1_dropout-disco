import torch
import torch.nn as nn


EMBED_DIM = 100  # dimensionality of the embedding vector (OG paper recommends 300)
EMBED_MAX_NORM = 1.0  # works as a regularisation parameter to prevent weights growing uncontrollably (?)


# inspired by github.com/OlgaChernytska/word2vec-pytorch
class CBOWModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        embed_max_norm: float = EMBED_MAX_NORM,
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            max_norm=embed_max_norm,
        )
        self.linear = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(inputs)
        bag = embeds.mean(dim=1)
        logits = self.linear(bag)
        return logits
