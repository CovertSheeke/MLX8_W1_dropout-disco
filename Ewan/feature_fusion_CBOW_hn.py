import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb
import numpy as np
import json
import os
import sys
from tqdm import tqdm

# --- Config ---
EMBEDDING_DIM = 100
TYPE_EMBED_DIM = 8
DAY_EMBED_DIM = 4
DOMAIN_EMBED_DIM = 16
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3
MAX_TITLE_TOKENS = 20  # truncate/pad titles to this length

TRAIN_PARQUET = "./.data/hn_posts_train_processed.parquet"
VAL_SPLIT = 0.1

# WORD2VEC_MODEL_PATH = "../word2vec/weights/word2vec_cbow_text8.pt"  # adjust if needed
# WORD2VEC_VOCAB_PATH = "../word2vec/weights/word2vec_cbow_text8_vocab.json"  # adjust if needed

# class Word2VecTitleEmbedder:
#     """
#     Loads a trained word2vec model and vocab, and provides embedding for a list of tokens.
#     """
#     def __init__(self, model_path=WORD2VEC_MODEL_PATH, vocab_path=WORD2VEC_VOCAB_PATH, device=None):
#         # Load vocab
#         with open(vocab_path, "r") as f:
#             self.vocab = json.load(f)
#         # Load model
#         self.model = torch.load(model_path, map_location="cpu")
#         self.embedding = self.model.embeddings  # nn.Embedding
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedding = self.embedding.to(self.device)
#         self.unk_idx = self.vocab.get("<UNK>", 0)
#
#     def encode(self, tokens):
#         idxs = [self.vocab.get(t, self.unk_idx) for t in tokens]
#         return torch.tensor(idxs, dtype=torch.long)
#
#     def __call__(self, idxs):
#         idxs = idxs.to(self.device)
#         return self.embedding(idxs)

# Replace the basic tokenisation with the project tokeniser
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../word2vec")))
# from tokeniser import tokenise

# --- Dataset ---
class HNRegressionDataset(Dataset):
    def __init__(self, parquet_path, word_embedder=None, max_title_tokens=MAX_TITLE_TOKENS):
        df = pd.read_parquet(parquet_path)
        # self.titles = df['title'].tolist()
        self.scores = df['score'].values.astype(np.float32)
        self.type_ids = df['type_id'].values.astype(np.int64)
        self.hour_of_day = df['hour_of_day'].values.astype(np.float32)
        self.day_of_week_id = df['day_of_week_id'].values.astype(np.int64)
        self.karma = df['karma'].values.astype(np.float32)
        self.descendants = df['descendants'].values.astype(np.float32)
        self.domain_id = df['domain_id'].values.astype(np.int64)
        # self.word_embedder = word_embedder
        # self.max_title_tokens = max_title_tokens

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        # Use the project tokeniser
        # tokens = tokenise(str(self.titles[idx]))
        # tokens = tokens[:self.max_title_tokens]
        # if len(tokens) < self.max_title_tokens:
        #     tokens += ["<UNK>"] * (self.max_title_tokens - len(tokens))
        # token_ids = self.word_embedder.encode(tokens)
        # Features
        features = {
            "type_id": self.type_ids[idx],
            "hour_of_day": self.hour_of_day[idx],
            "day_of_week_id": self.day_of_week_id[idx],
            "karma": self.karma[idx],
            "descendants": self.descendants[idx],
            "domain_id": self.domain_id[idx],
        }
        # return token_ids, features, self.scores[idx]
        return features, self.scores[idx]

def collate_fn(batch):
    # token_ids, features, targets = zip(*batch)
    # token_ids = torch.stack(token_ids)
    # targets = torch.tensor(targets, dtype=torch.float32)
    # # Stack features
    # features_stacked = {
    #     k: torch.tensor([f[k] for f in features], dtype=torch.long if "id" in k else torch.float32)
    #     for k in features[0]
    # }
    # return token_ids, features_stacked, targets
    features, targets = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.float32)
    features_stacked = {
        k: torch.tensor([f[k] for f in features], dtype=torch.long if "id" in k else torch.float32)
        for k in features[0]
    }
    return features_stacked, targets

# --- Model ---
class FeatureFusionRegressionModel(nn.Module):
    def __init__(
        self,
        # word_embedding_layer,
        type_vocab_size,
        day_vocab_size,
        domain_vocab_size,
        embedding_dim=EMBEDDING_DIM,
        type_embed_dim=TYPE_EMBED_DIM,
        day_embed_dim=DAY_EMBED_DIM,
        domain_embed_dim=DOMAIN_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()
        # self.word_embedding = word_embedding_layer
        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.day_embedding = nn.Embedding(day_vocab_size, day_embed_dim)
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(
                # embedding_dim + type_embed_dim + day_embed_dim + domain_embed_dim + 3,  # 3 numerical features
                type_embed_dim + day_embed_dim + domain_embed_dim + 3,  # 3 numerical features
                hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        # token_ids: [B, T]
        # embeds = self.word_embedding(token_ids)  # [B, T, D]
        # pooled = embeds.mean(dim=1)  # [B, D]
        type_emb = self.type_embedding(features["type_id"])
        day_emb = self.day_embedding(features["day_of_week_id"])
        domain_emb = self.domain_embedding(features["domain_id"])
        num_feats = torch.stack(
            [features["hour_of_day"], features["karma"], features["descendants"]], dim=1
        ).float()
        # x = torch.cat([pooled, type_emb, day_emb, domain_emb, num_feats], dim=1)
        x = torch.cat([type_emb, day_emb, domain_emb, num_feats], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)

# --- Trainer ---
def train_and_eval():
    wandb.init(project="feature-fusion-hn-upvotes", config={
        "embedding_dim": EMBEDDING_DIM,
        "type_embed_dim": TYPE_EMBED_DIM,
        "day_embed_dim": DAY_EMBED_DIM,
        "domain_embed_dim": DOMAIN_EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    })
    config = wandb.config

    # Load word2vec embedder
    # word_embedder = Word2VecTitleEmbedder(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Get vocab sizes from data (replace with actual max+1 if needed)
    df = pd.read_parquet(TRAIN_PARQUET)
    type_vocab_size = int(df["type_id"].max()) + 2
    day_vocab_size = int(df["day_of_week_id"].max()) + 2
    domain_vocab_size = int(df["domain_id"].max()) + 2

    # Split train/val
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    df_train.to_parquet("train_tmp.parquet")
    df_val.to_parquet("val_tmp.parquet")

    # train_ds = HNRegressionDataset("train_tmp.parquet", word_embedder)
    # val_ds = HNRegressionDataset("val_tmp.parquet", word_embedder)
    train_ds = HNRegressionDataset("train_tmp.parquet")
    val_ds = HNRegressionDataset("val_tmp.parquet")
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureFusionRegressionModel(
        # word_embedder.embedding,
        type_vocab_size,
        day_vocab_size,
        domain_vocab_size,
        embedding_dim=config.embedding_dim,
        type_embed_dim=config.type_embed_dim,
        day_embed_dim=config.day_embed_dim,
        domain_embed_dim=config.domain_embed_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for features, targets in tqdm(train_dl, desc=f"Epoch {epoch} [train]"):
            # token_ids = token_ids.to(device)
            features = {k: v.to(device) for k, v in features.items()}
            targets = targets.to(device)
            optimizer.zero_grad()
            # preds = model(token_ids, features)
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for features, targets in tqdm(val_dl, desc=f"Epoch {epoch} [val]"):
                # token_ids = token_ids.to(device)
                features = {k: v.to(device) for k, v in features.items()}
                targets = targets.to(device)
                # preds = model(token_ids, features)
                preds = model(features)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train_and_eval()
