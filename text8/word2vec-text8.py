import argparse
import os
import random
import sys
import zipfile
from collections import Counter
from datetime import datetime

import pandas as pd
import psutil
import platform
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

TEXT8_URL = os.getenv("TEXT8_URL", "http://mattmahoney.net/dc/text8.zip")
TEXT8_ZIP_PATH = os.getenv("TEXT8_ZIP_PATH", ".data/text8.zip")
TEXT8_RAW_PATH = os.getenv("TEXT8_RAW_PATH", ".data/text8")
MIN_COUNT = int(os.getenv("MIN_COUNT", "5"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "200"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "65536"))
EPOCHS = int(os.getenv("EPOCHS", "5"))
SGNS_MODEL_NAME = os.getenv("SGNS_MODEL_NAME", "sgns")
CBOW_MODEL_NAME = os.getenv("CBOW_MODEL_NAME", "cbow")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "text8-word2vec")
WANDB_RUN_NAME = os.getenv("WANDB_RUN_NAME", "text8-sgns-cbow")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", ".data/text8_compare.pt")
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.0001"))
SUBSAMPLING_THRESHOLD = float(os.getenv("SUBSAMPLING_THRESHOLD", "1e-5"))
NUMBER_WORKERS = int(os.getenv("NUMBER_WORKERS", "8"))
SHUFFLE = os.getenv("SHUFFLE", "True").lower() in ("1", "true", "yes")
PIN_MEMORY = os.getenv("PIN_MEMORY", "True").lower() in ("1", "true", "yes")
TOP_K = int(os.getenv("TOP_K", "10"))

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
def train(model, dataloader, epochs, device, wandb_run=None, model_name="", idx2word=None, word2idx=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.to(device)
    model.train()
    
    # Enable cudnn benchmarking for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(1, epochs+1):
        total_loss = 0
        total_pos_score = 0
        total_neg_score = 0
        total_batches = 0
        total_samples = 0
        
        # Enhanced progress bar with postfix updates
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            batch = [b.to(device, non_blocking=True) for b in batch]
            batch_size = batch[0].size(0)
            total_samples += batch_size
            
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
            
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss * batch_size
            total_pos_score += avg_pos * batch_size
            total_neg_score += avg_neg * batch_size
            total_batches += 1
            
            # Update progress bar with current metrics
            progress_bar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Pos': f'{avg_pos:.4f}',
                'Neg': f'{avg_neg:.4f}',
                'Avg Loss': f'{total_loss/total_samples:.4f}'
            })
            
            # Log every 100 batches for more frequent monitoring
            if wandb_run is not None and batch_idx % 100 == 0:
                # Log embedding norms for monitoring overfitting
                if hasattr(model, "target_emb"):
                    emb_norm = model.target_emb.weight.norm(dim=1).mean().item()
                    wandb_run.log({f"{model_name}_target_emb_norm_mean": emb_norm, "epoch": epoch, "batch_idx": batch_idx})
                if hasattr(model, "context_emb"):
                    emb_norm = model.context_emb.weight.norm(dim=1).mean().item()
                    wandb_run.log({f"{model_name}_context_emb_norm_mean": emb_norm, "epoch": epoch, "batch_idx": batch_idx})
                wandb_run.log({
                    f"{model_name}_batch_loss": batch_loss,
                    f"{model_name}_batch_pos_score": avg_pos,
                    f"{model_name}_batch_neg_score": avg_neg,
                    f"{model_name}_avg_loss": total_loss/total_samples,
                    "epoch": epoch,
                    "batch_idx": batch_idx
                })
        
        # Calculate epoch averages
        avg_loss = total_loss/total_samples
        avg_pos_score = total_pos_score/total_samples
        avg_neg_score = total_neg_score/total_samples
        print(f"Epoch {epoch} — Loss: {avg_loss:.4f} — Pos: {avg_pos_score:.4f} — Neg: {avg_neg_score:.4f}")
        
        if wandb_run is not None:
            # Log epoch-level embedding norms
            if hasattr(model, "target_emb"):
                emb_norm = model.target_emb.weight.norm(dim=1).mean().item()
                wandb_run.log({f"{model_name}_target_emb_norm_mean_epoch": emb_norm, "epoch": epoch})
            if hasattr(model, "context_emb"):
                emb_norm = model.context_emb.weight.norm(dim=1).mean().item()
                wandb_run.log({f"{model_name}_context_emb_norm_mean_epoch": emb_norm, "epoch": epoch})
            wandb_run.log({
                f"{model_name}_loss": avg_loss,
                f"{model_name}_pos_score": avg_pos_score,
                f"{model_name}_neg_score": avg_neg_score,
                "epoch": epoch
            })
            # --- Analogy metric: king-queen ≈ man-woman ---
            if idx2word is not None and word2idx is not None:
                # Use the correct embedding matrix for each model
                if model_name == "sgns":
                    emb_matrix = model.target_emb.weight.data
                elif model_name == "cbow":
                    emb_matrix = model.context_emb.weight.data
                else:
                    emb_matrix = None
                if emb_matrix is not None:
                    try:
                        analogy_sim = analogy_similarity(
                            emb_matrix, "king", "queen", "man", "woman", word2idx
                        )
                        wandb_run.log({f"{model_name}_analogy_king_queen_man_woman": analogy_sim, "epoch": epoch})
                    except Exception:
                        pass

def most_similar(emb_matrix, word, word2idx, idx2word, k=TOP_K):
    sims = F.cosine_similarity(
        emb_matrix[word2idx[word]].unsqueeze(0), emb_matrix)
    vals, idxs = sims.topk(k+1)
    results = [(idx2word[i], v) for i, v in zip(idxs.tolist(), vals.tolist()) if idx2word[i] != word][:k]
    print(f"Most similar to '{word}':")
    for w, score in results:
        print(f"  {w:15s} {score:.4f}")
    return results

def analogy_vector_length(emb_matrix, word_a, word_b, word_c, word_d, word2idx):
    # Compute the vector: emb(word_a) - emb(word_b) + emb(word_c) - emb(word_d)
    vec = emb_matrix[word2idx[word_a]] - emb_matrix[word2idx[word_b]] + emb_matrix[word2idx[word_c]] - emb_matrix[word2idx[word_d]]
    length = torch.norm(vec).item()
    print(f"Vector length for '{word_a} - {word_b} + {word_c} - {word_d}': {length:.6f}")
    return length

def evaluate_similarity_dataframe(emb_matrix, model_name, word2idx, idx2word, anchors, topk=10):
    results = []
    for anchor in anchors:
        if anchor not in word2idx:
            continue
        anchor_idx = word2idx[anchor]
        anchor_emb = emb_matrix[anchor_idx].unsqueeze(0)
        sims = F.cosine_similarity(anchor_emb, emb_matrix)
        vals, idxs = sims.topk(topk+1)  # include self
        filtered = [(idx2word[i], v) for i, v in zip(idxs.tolist(), vals.tolist()) if idx2word[i] != anchor]
        filtered = filtered[:topk]
        for word, score in filtered:
            results.append({"model": model_name, "anchor": anchor, "closest_word": word, "similarity": score})
    df = pd.DataFrame(results)
    return df

def word2vec_tests(model_path):
    print("Loading model checkpoint...")
    checkpoint = torch.load(model_path)
    sgns_emb = checkpoint['sgns_emb']
    cbow_emb = checkpoint['cbow_emb']
    word2idx = checkpoint['word2idx']
    idx2word = checkpoint['idx2word']

    print("SGNS Model Test:")
    most_similar(sgns_emb, 'king', word2idx, idx2word, TOP_K)
    most_similar(sgns_emb, 'queen', word2idx, idx2word, TOP_K)

    print("CBOW Model Test:")
    most_similar(cbow_emb, 'king', word2idx, idx2word, TOP_K)
    most_similar(cbow_emb, 'queen', word2idx, idx2word, TOP_K)

    print("SGNS Model Analogy Test:")
    def analogy(emb_matrix, word_a, word_b, word_c, word2idx, idx2word, k=TOP_K):
        # Check if all words exist in vocabulary
        for word in [word_a, word_b, word_c]:
            if word not in word2idx:
                print(f"Warning: Word '{word}' not found in vocabulary")
                return []
        vec = emb_matrix[word2idx[word_a]] - emb_matrix[word2idx[word_b]] + emb_matrix[word2idx[word_c]]
        sims = F.cosine_similarity(vec.unsqueeze(0), emb_matrix)
        vals, idxs = sims.topk(k+3)
        exclude = {word_a, word_b, word_c}
        result = []
        for i, v in zip(idxs.tolist(), vals.tolist()):
            if idx2word[i] not in exclude:
                result.append((idx2word[i], v))
            if len(result) == k:
                break
        print(f"Analogy '{word_a} - {word_b} + {word_c}' results:")
        for w, score in result:
            print(f"  {w:15s} {score:.4f}")
        return result

    analogy(sgns_emb, 'king', 'man', 'woman', word2idx, idx2word, TOP_K)
    print("CBOW Model Analogy Test:")
    analogy(cbow_emb, 'king', 'man', 'woman', word2idx, idx2word, TOP_K)

    print("SGNS Model Vector Length:")
    analogy_vector_length(sgns_emb, 'king', 'man', 'woman', 'queen', word2idx)
    print("CBOW Model Vector Length:")
    analogy_vector_length(cbow_emb, 'king', 'man', 'woman', 'queen', word2idx)

    # New evaluation: build dataframes of closest words for a list of anchors
    anchors = ["king", "queen", "lawyer", "apple", "cat", "ear", "animal",
               "city", "clothing", "color", "country", "emotion", "fruit",
               "professional", "technology"]
    df_sgns = evaluate_similarity_dataframe(sgns_emb, "sgns", word2idx, idx2word, anchors, topk=10)
    df_cbow = evaluate_similarity_dataframe(cbow_emb, "cbow", word2idx, idx2word, anchors, topk=10)
    df_all = pd.concat([df_sgns, df_cbow], ignore_index=True)
    print("\nEvaluation DataFrame (each row = one closest word result):")
    
    # Display all records without truncation
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        print(df_all)

def analogy_similarity(emb_matrix, word_a, word_b, word_c, word_d, word2idx):
    # Check if all words exist in vocabulary
    for word in [word_a, word_b, word_c, word_d]:
        if word not in word2idx:
            raise KeyError(f"Word '{word}' not found in vocabulary")
    # Compute cosine similarity between (a-b) and (c-d)
    vec1 = emb_matrix[word2idx[word_a]] - emb_matrix[word2idx[word_b]]
    vec2 = emb_matrix[word2idx[word_c]] - emb_matrix[word2idx[word_d]]
    sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    return sim

def print_system_info():
    print("=== System Info ===")
    # CPU info
    try:
        print("Platform:", platform.platform())
        print("Processor:", platform.processor())
        print("CPU count (logical):", psutil.cpu_count(logical=True))
        print("CPU count (physical):", psutil.cpu_count(logical=False))
        print("Total RAM (GB):", round(psutil.virtual_memory().total / (1024**3), 2))
    except Exception as e:
        print("Could not get CPU/memory info:", e)
    # GPU info
    if torch.cuda.is_available():
        try:
            gpu_idx = torch.cuda.current_device()
            print("CUDA device count:", torch.cuda.device_count())
            print("CUDA device name:", torch.cuda.get_device_name(gpu_idx))
            print("CUDA capability:", torch.cuda.get_device_capability(gpu_idx))
            print("CUDA memory total (GB):", round(torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3), 2))
            print("CUDA memory allocated (GB):", round(torch.cuda.memory_allocated(gpu_idx) / (1024**3), 2))
            print("CUDA memory reserved (GB):", round(torch.cuda.memory_reserved(gpu_idx) / (1024**3), 2))
        except Exception as e:
            print("Could not get CUDA info:", e)
    else:
        print("CUDA not available.")

def main():
    parser = argparse.ArgumentParser(
        description="Train word2vec (SGNS/CBOW) on text8. Use --download to fetch text8, --train to train models."
    )
    parser.add_argument("--download", action="store_true", help="Download and extract text8 dataset")
    parser.add_argument("--train", action="store_true", help="Train SGNS and CBOW models")
    parser.add_argument("--test", action="store_true", help="Run word2vec_tests on saved model")
    args = parser.parse_args()

    # Warn if .env does not exist
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        print(f"Warning: .env file not found at {env_path}. Using default environment variables.")

    # Check if text8 exists
    text8_exists = os.path.isfile(TEXT8_RAW_PATH)

    if not (args.download or args.train or args.test):
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
        
        # Add system info printing
        print_system_info()
        
        print("Loading corpus... this may take a while.")
        raw_path = TEXT8_RAW_PATH
        tokens = load_tokens(raw_path)
        counter = Counter(tokens)
        vocab = {w: i for i, (w, c) in enumerate(counter.items()) if c >= MIN_COUNT}
        idx2word = list(vocab.keys())
        word2idx = {w: i for i, w in enumerate(idx2word)}
        freqs = torch.Tensor([counter[w] for w in idx2word])
        # subsampling - use configurable threshold
        tau = SUBSAMPLING_THRESHOLD * len(tokens)
        probs = ((freqs / tau).sqrt() + 1) * (tau / freqs)
        probs = torch.clamp(probs, max=1.0)
        
        # Add timestamp to WANDB_RUN_NAME
        now_str = datetime.now().strftime("_%Y%m%d%H%M")
        run_name = WANDB_RUN_NAME + now_str

        wandb.init(project=WANDB_PROJECT, name=run_name, config={
            "embedding_dim": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "min_count": MIN_COUNT,
            "learning_rate": LEARNING_RATE,
            "subsampling_threshold": SUBSAMPLING_THRESHOLD
        })
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # SGNS
        sgns_ds = SGNSDataset(tokens, word2idx, probs)
        print(f"SGNS DataLoader params: batch_size={BATCH_SIZE}, shuffle={SHUFFLE}, num_workers={NUMBER_WORKERS}, pin_memory={PIN_MEMORY}")
        sgns_loader = DataLoader(
            sgns_ds,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUMBER_WORKERS,
            pin_memory=PIN_MEMORY
        )
        sgns_model = SGNS(len(idx2word), emb_size=EMBEDDING_DIM)
        
        # Print model info
        total_params = sum(p.numel() for p in sgns_model.parameters())
        print(f"SGNS model - Total parameters: {total_params:,}")
        
        print("Training SGNS...")
        train(
            sgns_model, sgns_loader, epochs=EPOCHS, device=device,
            wandb_run=wandb, model_name=SGNS_MODEL_NAME,
            idx2word=idx2word, word2idx=word2idx
        )

        # CBOW
        cbow_ds = CBOWDataset(tokens, word2idx, probs)
        print(f"CBOW DataLoader params: batch_size={BATCH_SIZE}, shuffle={SHUFFLE}, num_workers={NUMBER_WORKERS}, pin_memory={PIN_MEMORY}")
        cbow_loader = DataLoader(
            cbow_ds,
            batch_size=BATCH_SIZE,
            shuffle=SHUFFLE,
            num_workers=NUMBER_WORKERS,
            pin_memory=PIN_MEMORY,
            collate_fn=lambda x: (
                nn.utils.rnn.pad_sequence([c for c, *_ in x], batch_first=True, padding_value=0),
                torch.tensor([t for _, t, _ in x]),
                torch.stack([n for *_, n in x])
            )
        )
        cbow_model = CBOW(len(idx2word), emb_size=EMBEDDING_DIM)
        
        # Print model info
        total_params = sum(p.numel() for p in cbow_model.parameters())
        print(f"CBOW model - Total parameters: {total_params:,}")
        
        print("Training CBOW...")
        train(
            cbow_model, cbow_loader, epochs=EPOCHS, device=device,
            wandb_run=wandb, model_name=CBOW_MODEL_NAME,
            idx2word=idx2word, word2idx=word2idx
        )

        print("Training complete. Saving models and embeddings...")
        torch.save({
            "sgns_emb": sgns_model.target_emb.weight.data.cpu(),
            "cbow_emb": cbow_model.context_emb.weight.data.cpu(),
            "word2idx": word2idx,
            "idx2word": idx2word
        }, OUTPUT_PATH)

        if args.test:
            word2vec_tests(OUTPUT_PATH)
        wandb.finish()
        sys.exit(0)

    if args.test and not args.train:
        if not os.path.isfile(OUTPUT_PATH):
            print(f"Model checkpoint not found at {OUTPUT_PATH}. Please train the model first.")
            sys.exit(1)
        word2vec_tests(OUTPUT_PATH)
        sys.exit(0)

    # If we get here, something went wrong
    parser.print_help()
    sys.exit(1)

# Setup and run
if __name__ == "__main__":
    main()
