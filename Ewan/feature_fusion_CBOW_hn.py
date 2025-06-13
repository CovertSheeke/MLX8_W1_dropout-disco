import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import wandb
import numpy as np
import os
import sys
from tqdm import tqdm

# --- Config ---
EMBEDDING_DIM = 100
DOMAIN_EMBED_DIM = 10
AUTHOR_EMBED_DIM = 10
HIDDEN_DIM = 128
BATCH_SIZE = 2048
EPOCHS = 10
LR = 1e-3

TRAIN_PARQUET = "./.data/hn_posts_train_processed.parquet"
VAL_SPLIT = 0.1

# --- Dataset ---
class HNRegressionDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        self.scores = df['score'].values.astype(np.float32)
        self.domain_ids = df['domain_id'].values.astype(np.int64)
        # Handle author_id if present, else encode as categorical
        if 'author_id' in df.columns:
            self.author_ids = df['author_id'].values.astype(np.int64)
        else:
            self.author_ids = df['author'].astype("category").cat.codes.values.astype(np.int64)

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        features = {
            "domain_id": self.domain_ids[idx],
            "author_id": self.author_ids[idx],
        }
        return features, self.scores[idx]

def collate_fn(batch):
    features, targets = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.float32)
    features_stacked = {
        k: torch.tensor([f[k] for f in features], dtype=torch.long)
        for k in features[0]
    }
    return features_stacked, targets

# --- Model ---
class FeatureFusionRegressionModel(nn.Module):
    def __init__(
        self,
        domain_vocab_size,
        author_vocab_size,
        domain_embed_dim=DOMAIN_EMBED_DIM,
        author_embed_dim=AUTHOR_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embed_dim)
        self.author_embedding = nn.Embedding(author_vocab_size, author_embed_dim)
        total_dim = domain_embed_dim + author_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        domain_emb = self.domain_embedding(features["domain_id"])
        author_emb = self.author_embedding(features["author_id"])
        x = torch.cat([domain_emb, author_emb], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)

# --- Trainer ---
def train_and_eval():
    wandb.init(project="feature-fusion-hn-upvotes", config={
        "domain_embed_dim": DOMAIN_EMBED_DIM,
        "author_embed_dim": AUTHOR_EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    })
    config = wandb.config

    # Get vocab sizes from data using max()+1 for correct embedding size
    df = pd.read_parquet(TRAIN_PARQUET)
    domain_vocab_size = int(df["domain_id"].max()) + 1
    author_vocab_size = int(df["author"].astype("category").cat.codes.max()) + 1

    # Split train/val
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1 - VAL_SPLIT))
    train_idx, val_idx = idx[:split], idx[split:]
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    # Ensure categorical encoding is consistent between train/val
    col = "author"
    cat = pd.Categorical(df[col])
    df_train[col] = pd.Categorical(df_train[col], categories=cat.categories).codes
    df_val[col] = pd.Categorical(df_val[col], categories=cat.categories).codes
    max_code = cat.codes.max()
    df_train.loc[df_train[col] == -1, col] = max_code
    df_val.loc[df_val[col] == -1, col] = max_code

    df_train.to_parquet("train_tmp.parquet")
    df_val.to_parquet("val_tmp.parquet")

    train_ds = HNRegressionDataset("train_tmp.parquet")
    val_ds = HNRegressionDataset("val_tmp.parquet")
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureFusionRegressionModel(
        domain_vocab_size,
        author_vocab_size,
        domain_embed_dim=config.domain_embed_dim,
        author_embed_dim=config.author_embed_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for features, targets in tqdm(train_dl, desc=f"Epoch {epoch} [train]"):
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
            targets = targets.to(device)
            # Debug: check for out-of-bounds indices
            for name, vocab_size in [
                ("domain_id", domain_vocab_size),
                ("author_id", author_vocab_size),
            ]:
                if torch.any(features[name] >= vocab_size) or torch.any(features[name] < 0):
                    print(f"Out-of-bounds index in {name}: min={features[name].min().item()}, max={features[name].max().item()}, vocab_size={vocab_size}")
                    raise ValueError(f"Out-of-bounds index in {name}")
            optimizer.zero_grad()
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
                features = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
                targets = targets.to(device)
                # Debug: check for out-of-bounds indices
                for name, vocab_size in [
                    ("domain_id", domain_vocab_size),
                    ("author_id", author_vocab_size),
                ]:
                    if torch.any(features[name] >= vocab_size) or torch.any(features[name] < 0):
                        print(f"Out-of-bounds index in {name}: min={features[name].min().item()}, max={features[name].max().item()}, vocab_size={vocab_size}")
                        raise ValueError(f"Out-of-bounds index in {name}")
                preds = model(features)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train_and_eval()
