import torch
import torch.nn as nn


EMBED_DIM = 100  # dimensionality of the embedding vector (OG paper recommends 300)
EMBED_MAX_NORM = 1.0  # works as a regularisation parameter to prevent weights growing uncontrollably (?)


# TODO: consider implementing negative sampling for speedup (as per original paper)
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
        # add in a hidden layer for 'non-linear transformational capability'
        self.hidden = nn.Linear(embed_dim, embed_dim)
        # add dropout for regularisation (to prevent overfitting)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(
            in_features=embed_dim,
            out_features=vocab_size,
        )
        # initialise weights more explicitly
        self._init_weights()

    def _init_weights(self):
        # init embeddings with small random values
        nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        # init linear layers (hidden + output) with Glorot uniform (Xavier) initialisation
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.linear.weight)
        # init biases to zero, as is standard
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.linear.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(inputs)
        bag = embeds.mean(dim=1)
        # crucially, we use a ReLU activation function to introduce non-linearity
        hidden = torch.relu(self.hidden(bag))
        hidden = self.dropout(hidden)
        logits = self.linear(hidden)
        return logits
