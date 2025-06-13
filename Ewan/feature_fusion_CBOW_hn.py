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
TYPE_EMBED_DIM = 8
DAY_EMBED_DIM = 4
DOMAIN_EMBED_DIM = 16
AUTHOR_EMBED_DIM = 16
USER_EMBED_DIM = 16
ITEM_EMBED_DIM = 8
HIDDEN_DIM = 128
BATCH_SIZE = 128
EPOCHS = 10
LR = 1e-3

TRAIN_PARQUET = "./.data/hn_posts_train_processed.parquet"
VAL_SPLIT = 0.1

# --- Dataset ---
class HNRegressionDataset(Dataset):
    def __init__(self, parquet_path):
        df = pd.read_parquet(parquet_path)
        self.scores = df['score'].values.astype(np.float32)
        self.type_ids = df['type_id'].values.astype(np.int64)
        self.hour_of_day = df['hour_of_day'].values.astype(np.float32)
        self.day_of_week_ids = df['day_of_week_id'].values.astype(np.int64)
        self.karma = df['karma'].values.astype(np.float32)
        self.descendants = df['descendants'].values.astype(np.float32)
        self.domain_ids = df['domain_id'].values.astype(np.int64)
        # Handle author_id and user_id if present, else encode as categorical
        if 'author_id' in df.columns:
            self.author_ids = df['author_id'].values.astype(np.int64)
        else:
            self.author_ids = df['author'].astype("category").cat.codes.values.astype(np.int64)
        if 'user_id' in df.columns and df['user_id'].dtype in [np.int64, np.int32]:
            self.user_ids = df['user_id'].values.astype(np.int64)
        else:
            self.user_ids = df['user_id'].astype("category").cat.codes.values.astype(np.int64)
        if 'item_id' in df.columns:
            self.item_ids = df['item_id'].values.astype(np.int64)
        else:
            self.item_ids = np.zeros(len(df), dtype=np.int64)
        self.titles = df['title'].astype(str).tolist()
        self.urls = df['url'].astype(str).tolist()

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        features = {
            "type_id": self.type_ids[idx],
            "hour_of_day": self.hour_of_day[idx],
            "day_of_week_id": self.day_of_week_ids[idx],
            "karma": self.karma[idx],
            "descendants": self.descendants[idx],
            "domain_id": self.domain_ids[idx],
            "author_id": self.author_ids[idx],
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "title": self.titles[idx],
            "url": self.urls[idx],
        }
        return features, self.scores[idx]

def collate_fn(batch):
    features, targets = zip(*batch)
    targets = torch.tensor(targets, dtype=torch.float32)
    features_stacked = {
        k: torch.tensor([f[k] for f in features], dtype=torch.long if "id" in k else torch.float32)
        for k in features[0]
        if k not in ["title", "url"]
    }
    # Optionally keep titles/urls as list of strings for further processing
    features_stacked["title"] = [f["title"] for f in features]
    features_stacked["url"] = [f["url"] for f in features]
    return features_stacked, targets

# --- Model ---
class FeatureFusionRegressionModel(nn.Module):
    def __init__(
        self,
        type_vocab_size,
        day_vocab_size,
        domain_vocab_size,
        author_vocab_size,
        user_vocab_size,
        item_vocab_size,
        type_embed_dim=TYPE_EMBED_DIM,
        day_embed_dim=DAY_EMBED_DIM,
        domain_embed_dim=DOMAIN_EMBED_DIM,
        author_embed_dim=AUTHOR_EMBED_DIM,
        user_embed_dim=USER_EMBED_DIM,
        item_embed_dim=ITEM_EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
    ):
        super().__init__()
        self.type_embedding = nn.Embedding(type_vocab_size, type_embed_dim)
        self.day_embedding = nn.Embedding(day_vocab_size, day_embed_dim)
        self.domain_embedding = nn.Embedding(domain_vocab_size, domain_embed_dim)
        self.author_embedding = nn.Embedding(author_vocab_size, author_embed_dim)
        self.user_embedding = nn.Embedding(user_vocab_size, user_embed_dim)
        self.item_embedding = nn.Embedding(item_vocab_size, item_embed_dim)
        # 3 numerical features: hour_of_day, karma, descendants
        total_dim = (
            type_embed_dim + day_embed_dim + domain_embed_dim +
            author_embed_dim + user_embed_dim + item_embed_dim + 3
        )
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        type_emb = self.type_embedding(features["type_id"])
        day_emb = self.day_embedding(features["day_of_week_id"])
        domain_emb = self.domain_embedding(features["domain_id"])
        author_emb = self.author_embedding(features["author_id"])
        user_emb = self.user_embedding(features["user_id"])
        item_emb = self.item_embedding(features["item_id"])
        num_feats = torch.stack(
            [features["hour_of_day"].float(), features["karma"].float(), features["descendants"].float()], dim=1
        )
        x = torch.cat([type_emb, day_emb, domain_emb, author_emb, user_emb, item_emb, num_feats], dim=1)
        out = self.mlp(x)
        return out.squeeze(-1)

# --- Trainer ---
def train_and_eval():
    wandb.init(project="feature-fusion-hn-upvotes", config={
        "type_embed_dim": TYPE_EMBED_DIM,
        "day_embed_dim": DAY_EMBED_DIM,
        "domain_embed_dim": DOMAIN_EMBED_DIM,
        "author_embed_dim": AUTHOR_EMBED_DIM,
        "user_embed_dim": USER_EMBED_DIM,
        "item_embed_dim": ITEM_EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    })
    config = wandb.config

    # Get vocab sizes from data
    df = pd.read_parquet(TRAIN_PARQUET)
    type_vocab_size = int(df["type_id"].max()) + 2
    day_vocab_size = int(df["day_of_week_id"].max()) + 2
    domain_vocab_size = int(df["domain_id"].max()) + 2
    author_vocab_size = int(df["author"].astype("category").cat.codes.max()) + 2
    user_vocab_size = int(df["user_id"].astype("category").cat.codes.max()) + 2
    item_vocab_size = int(df["item_id"].max()) + 2 if "item_id" in df.columns else 2

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

    train_ds = HNRegressionDataset("train_tmp.parquet")
    val_ds = HNRegressionDataset("val_tmp.parquet")
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureFusionRegressionModel(
        type_vocab_size,
        day_vocab_size,
        domain_vocab_size,
        author_vocab_size,
        user_vocab_size,
        item_vocab_size,
        type_embed_dim=config.type_embed_dim,
        day_embed_dim=config.day_embed_dim,
        domain_embed_dim=config.domain_embed_dim,
        author_embed_dim=config.author_embed_dim,
        user_embed_dim=config.user_embed_dim,
        item_embed_dim=config.item_embed_dim,
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
                preds = model(features)
                loss = criterion(preds, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    train_and_eval()
