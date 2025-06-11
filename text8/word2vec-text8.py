import os
import zipfile
import requests
import random
from collections import Counter
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

TEXT8_URL = os.getenv("TEXT8_URL", "http://mattmahoney.net/dc/text8.zip")
TEXT8_ZIP_PATH = os.getenv("TEXT8_ZIP_PATH", ".data/text8.zip")
TEXT8_RAW_PATH = os.getenv("TEXT8_RAW_PATH", ".data/text8")
MIN_COUNT = int(os.getenv("MIN_COUNT", "5"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
EPOCHS = int(os.getenv("EPOCHS", "3"))
SGNS_MODEL_NAME = os.getenv("SGNS_MODEL_NAME", "sgns")
CBOW_MODEL_NAME = os.getenv("CBOW_MODEL_NAME", "cbow")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "text8-word2vec")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "text8-sgns-cbow")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", ".data/text8_compare.pt")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))

# Ensure .data directory exists
data_dir = os.path.dirname(TEXT8_ZIP_PATH) or "."
os.makedirs(data_dir, exist_ok=True)

# 1. Download & unpack Text8 if needed
def download_text8(url=TEXT8_URL, zip_path=TEXT8_ZIP_PATH, raw_path=TEXT8_RAW_PATH):
    if not os.path.isfile(raw_path):
        if not os.path.isfile(zip_path):
            print(f"Downloading text8 from {url} into {zip_path} ...")
            resp = requests.get(url)
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
            with open(zip_path, "wb") as f:
                f.write(resp.content)
        print(f"Unzipping {zip_path} into {os.path.dirname(raw_path) or '.'} ...")
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(raw_path) or ".", exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(os.path.dirname(raw_path) or ".")
    return raw_path

# 2. Read and tokenize
def load_tokens(path):
    with open(path, "r") as f:
        return f.read().split()

    # 3. Build vocab and subsampling
    min_count = MIN_COUNT
    print("Loading corpus... this may take a while.")
    raw_path = download_text8()
    tokens = load_tokens(raw_path)
    counter = Counter(tokens)
    vocab = {w: i for i, (w, c) in enumerate(counter.items()) if c >= min_count}
    idx2word = list(vocab.keys())
    word2idx = {w: i for i, w in enumerate(idx2word)}
    freqs = torch.Tensor([counter[w] for w in idx2word])
    # subsampling
    τ = 1e-5 * len(tokens)
    probs = ((freqs / τ).sqrt() + 1) * (τ / freqs)
    probs = torch.clamp(probs, max=1.0)

# Dataset for SGNS
class SGNSDataset(Dataset):
    def __init__(self, tokens, word2idx, probs, window_size=5, neg_samples=5):
        self.tokens = [word2idx[w] for w in tokens if w in word2idx]
        self.probs = probs
        self.vocab_size = len(word2idx)
        self.window_size = window_size
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        target = self.tokens[idx]
        if random.random() > self.probs[target]:
            return self.__getitem__(random.randint(0, len(self.tokens)-1))
        window = random.randint(1, self.window_size)
        start = max(0, idx - window)
        end = min(len(self.tokens), idx + window + 1)
        contexts = [self.tokens[i] for i in range(start, end) if i != idx]
        if not contexts:
            return self.__getitem__(random.randint(0, len(self.tokens)-1))
        context = random.choice(contexts)
        negs = []
        while len(negs) < self.neg_samples:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                negs.append(neg)
        return target, context, torch.LongTensor(negs)

# Dataset for CBOW
class CBOWDataset(Dataset):
    def __init__(self, tokens, word2idx, probs, window_size=5, neg_samples=5):
        self.tokens = [word2idx[w] for w in tokens if w in word2idx]
        self.probs = probs
        self.vocab_size = len(word2idx)
        self.window_size = window_size
        self.neg_samples = neg_samples

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        target = self.tokens[idx]
        if random.random() > self.probs[target]:
            return self.__getitem__(random.randint(0, len(self.tokens)-1))
        window = random.randint(1, self.window_size)
        start = max(0, idx - window)
        end = min(len(self.tokens), idx + window + 1)
        context_idxs = [self.tokens[i] for i in range(start, end) if i != idx]
        if not context_idxs:
            return self.__getitem__(random.randint(0, len(self.tokens)-1))
        # pad contexts to fixed size for batching
        # here we simply use all available contexts
        negs = []
        while len(negs) < self.neg_samples:
            neg = random.randint(0, self.vocab_size - 1)
            if neg != target:
                negs.append(neg)
        return torch.LongTensor(context_idxs), target, torch.LongTensor(negs)

# Models
class SGNS(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.target_emb = nn.Embedding(vocab_size, emb_size)
        self.context_emb = nn.Embedding(vocab_size, emb_size)
        nn.init.xavier_uniform_(self.target_emb.weight)
        nn.init.xavier_uniform_(self.context_emb.weight)

    def forward(self, targets, contexts, negatives):
        v_t = self.target_emb(targets)
        v_c = self.context_emb(contexts)
        v_n = self.context_emb(negatives)
        pos_score = torch.sum(v_t * v_c, dim=1)
        neg_score = torch.bmm(v_n, v_t.unsqueeze(2)).squeeze()
        loss = -torch.log(torch.sigmoid(pos_score)).mean() \
               - torch.log(torch.sigmoid(-neg_score)).sum(1).mean()
        return loss

class CBOW(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.context_emb = nn.Embedding(vocab_size, emb_size)
        self.target_emb = nn.Embedding(vocab_size, emb_size)
        nn.init.xavier_uniform_(self.context_emb.weight)
        nn.init.xavier_uniform_(self.target_emb.weight)

    def forward(self, contexts, targets, negatives):
        # contexts: [B, C]
        v_c = self.context_emb(contexts)           # [B, C, E]
        v_c_mean = v_c.mean(dim=1)                 # [B, E]
        v_t = self.target_emb(targets)             # [B, E]
        v_n = self.target_emb(negatives)           # [B, neg, E]
        pos_score = torch.sum(v_c_mean * v_t, dim=1)
        neg_score = torch.bmm(v_n, v_c_mean.unsqueeze(2)).squeeze()
        loss = -torch.log(torch.sigmoid(pos_score)).mean() \
               - torch.log(torch.sigmoid(-neg_score)).sum(1).mean()
        return loss

# Training utility
def train(model, dataloader, epochs, device, wandb_run=None, model_name=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0
        total_pos_score = 0
        total_neg_score = 0
        total_batches = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            batch = [b.to(device) for b in batch]
            # --- Compute loss and metrics ---
            if model_name == "sgns":
                targets, contexts, negatives = batch
                v_t = model.target_emb(targets)
                v_c = model.context_emb(contexts)
                v_n = model.context_emb(negatives)
                pos_score = torch.sum(v_t * v_c, dim=1)
                neg_score = torch.bmm(v_n, v_t.unsqueeze(2)).squeeze()
                loss = -torch.log(torch.sigmoid(pos_score)).mean() \
                       - torch.log(torch.sigmoid(-neg_score)).sum(1).mean()
                avg_pos = pos_score.mean().item()
                avg_neg = neg_score.mean().item()
            elif model_name == "cbow":
                contexts, targets, negatives = batch
                v_c = model.context_emb(contexts)
                v_c_mean = v_c.mean(dim=1)
                v_t = model.target_emb(targets)
                v_n = model.target_emb(negatives)
                pos_score = torch.sum(v_c_mean * v_t, dim=1)
                neg_score = torch.bmm(v_n, v_c_mean.unsqueeze(2)).squeeze()
                loss = -torch.log(torch.sigmoid(pos_score)).mean() \
                       - torch.log(torch.sigmoid(-neg_score)).sum(1).mean()
                avg_pos = pos_score.mean().item()
                avg_neg = neg_score.mean().item()
            else:
                loss = model(*batch)
                avg_pos = avg_neg = 0
            # --- End metrics ---
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_pos_score += avg_pos
            total_neg_score += avg_neg
            total_batches += 1
            if wandb_run is not None:
                wandb_run.log({
                    f"{model_name}_batch_loss": loss.item(),
                    f"{model_name}_batch_pos_score": avg_pos,
                    f"{model_name}_batch_neg_score": avg_neg,
                    "epoch": epoch
                })
        avg_loss = total_loss/total_batches
        avg_pos_score = total_pos_score/total_batches
        avg_neg_score = total_neg_score/total_batches
        print(f"Epoch {epoch} — Loss: {avg_loss:.4f} — Pos: {avg_pos_score:.4f} — Neg: {avg_neg_score:.4f}")
        if wandb_run is not None:
            wandb_run.log({
                f"{model_name}_loss": avg_loss,
                f"{model_name}_pos_score": avg_pos_score,
                f"{model_name}_neg_score": avg_neg_score,
                "epoch": epoch
            })

def most_similar(emb_matrix, word, word2idx, idx2word, k=10):
    sims = F.cosine_similarity(
        emb_matrix[word2idx[word]].unsqueeze(0), emb_matrix)
    vals, idxs = sims.topk(k+1)
    return [(idx2word[i], v.item()) for i, v in zip(idxs.tolist(), vals.tolist()) if idx2word[i] != word][:k]

def word2vec_tests(model_path):
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path)
    sgns_emb = checkpoint['sgns_emb']
    cbow_emb = checkpoint['cbow_emb']
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    print("SGNS top 5 similar to 'king':", most_similar(sgns_emb, 'king', word2idx, idx2word, 5))
    print("CBOW top 5 similar to 'king':", most_similar(cbow_emb, 'king', word2idx, idx2word, 5))

    # king - man + woman = ?
    def analogy(emb_matrix, word_a, word_b, word_c, word2idx, idx2word, k=5):
        vec = emb_matrix[word2idx[word_a]] - emb_matrix[word2idx[word_b]] + emb_matrix[word2idx[word_c]]
        sims = F.cosine_similarity(vec.unsqueeze(0), emb_matrix)
        vals, idxs = sims.topk(k+3)
        # Exclude input words from results
        exclude = {word_a, word_b, word_c}
        result = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            if idx2word[i] not in exclude:
                result.append((idx2word[i], v.item()))
            if len(result) == k:
                break
        return result

    print("SGNS analogy 'king - man + woman':", analogy(sgns_emb, 'king', 'man', 'woman', word2idx, idx2word, 5))
    print("CBOW analogy 'king - man + woman':", analogy(cbow_emb, 'king', 'man', 'woman', word2idx, idx2word, 5))

def main():
    parser = argparse.ArgumentParser(
        description="Train word2vec (SGNS/CBOW) on text8. Use --download to fetch text8, --train to train models."
    )
    parser.add_argument("--download", action="store_true", help="Download and extract text8 dataset")
    parser.add_argument("--train", action="store_true", help="Train SGNS and CBOW models")
    args = parser.parse_args()

    # Warn if .env does not exist
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        print(f"Warning: .env file not found at {env_path}. Using default environment variables.")

    # Check if text8 exists
    text8_exists = os.path.isfile(TEXT8_RAW_PATH)

    if not (args.download or args.train):
        parser.print_help()
        sys.exit(0)

    if args.download:
        if text8_exists:
            resp = input(f"File {TEXT8_RAW_PATH} already exists. Overwrite? [y/N] ").strip().lower()
            if resp != "y":
                print("Aborting download.")
                sys.exit(0)
            else:
                os.remove(TEXT8_RAW_PATH)
        download_text8()
        print(f"Downloaded and extracted text8 to {TEXT8_RAW_PATH}")
        sys.exit(0)

    if args.train:
        if not text8_exists:
            print(f"Text8 file not found at {TEXT8_RAW_PATH}. Please run with --download first.")
            sys.exit(1)
        print("Loading corpus... this may take a while.")
        raw_path = TEXT8_RAW_PATH
        tokens = load_tokens(raw_path)
        counter = Counter(tokens)
        vocab = {w: i for i, (w, c) in enumerate(counter.items()) if c >= MIN_COUNT}
        idx2word = list(vocab.keys())
        word2idx = {w: i for i, w in enumerate(idx2word)}
        freqs = torch.Tensor([counter[w] for w in idx2word])
        # subsampling
        τ = 1e-5 * len(tokens)
        probs = ((freqs / τ).sqrt() + 1) * (τ / freqs)
        probs = torch.clamp(probs, max=1.0)

        # Add timestamp to WANDB_RUN_NAME
        now_str = datetime.now().strftime("_%Y%m%d%H%M")
        run_name = WANDB_RUN_NAME + now_str

        wandb.init(project=WANDB_PROJECT, name=run_name, config={
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "min_count": MIN_COUNT
        })
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # SGNS
        sgns_ds = SGNSDataset(tokens, word2idx, probs)
        sgns_loader = DataLoader(sgns_ds, batch_size=BATCH_SIZE, shuffle=True)
        sgns_model = SGNS(len(idx2word), emb_size=EMBEDDING_DIM)
        print("Training SGNS...")
        train(sgns_model, sgns_loader, epochs=EPOCHS, device=device, wandb_run=wandb, model_name=SGNS_MODEL_NAME)

        # CBOW
        cbow_ds = CBOWDataset(tokens, word2idx, probs)
        cbow_loader = DataLoader(cbow_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: (
            nn.utils.rnn.pad_sequence([c for c, *_ in x], batch_first=True, padding_value=0),
            torch.tensor([t for _, t, _ in x]),
            torch.stack([n for *_, n in x])
        ))
        cbow_model = CBOW(len(idx2word), emb_size=EMBEDDING_DIM)
        print("Training CBOW...")
        train(cbow_model, cbow_loader, epochs=EPOCHS, device=device, wandb_run=wandb, model_name=CBOW_MODEL_NAME)

        print("Training complete. Saving models and embeddings...")
        torch.save({
            "sgns_emb": sgns_model.target_emb.weight.data.cpu(),
            "cbow_emb": cbow_model.context_emb.weight.data.cpu(),
            "word2idx": word2idx,
            "idx2word": idx2word
        }, OUTPUT_PATH)

        word2vec_tests(OUTPUT_PATH)
        wandb.finish()
        sys.exit(0)

    # If we get here, something went wrong
    parser.print_help()
    sys.exit(1)

# Setup and run
if __name__ == "__main__":
    main()
