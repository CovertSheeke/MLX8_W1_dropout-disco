import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from dotenv import load_dotenv
from tqdm import tqdm
import wandb
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, max_error
from scipy.stats import pearsonr, spearmanr

# Load environment variables
load_dotenv()

# Paths from environment variables
TRAIN_FILE = os.getenv("PROCESSED_TRAIN_FILE", "../postgresql/.data/hn_posts_train_processed.parquet")
TEST_FILE = os.getenv("PROCESSED_TEST_FILE", "../postgresql/.data/hn_posts_test_processed.parquet")
WORD2VEC_PATH = os.getenv("WORD2VEC_PATH", "../text8/.data/text8_compare.pt")
FUSION_MODEL_SAVE_PATH = os.getenv("FUSION_MODEL_SAVE_PATH", "./.data/fusion_model.pt")

# Hyperparameters (example values; see .env.example for full list)
BATCH_SIZE = int(os.getenv("FUSION_BATCH_SIZE", "8192"))
FUSION_EPOCHS = int(os.getenv("FUSION_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("FUSION_LEARNING_RATE", "0.001"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "200"))  # should match word2vec dims

# Embedding sizes for categorical features
TYPE_EMB_DIM = int(os.getenv("TYPE_EMB_DIM", "8"))
DAY_EMB_DIM = int(os.getenv("DAY_EMB_DIM", "3"))
DOMAIN_EMB_DIM = int(os.getenv("DOMAIN_EMB_DIM", "8"))

# Fusion network hidden layer size
FC_HIDDEN_SIZE = int(os.getenv("FC_HIDDEN_SIZE", "64"))

# Simple tokenizer function
def simple_tokenize(text):
    text = text.lower() if isinstance(text, str) else ""
    # Remove punctuation; keep only alphanumerics and spaces
    import re
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.split()

# Fusion Dataset: loads processed parquet and produces:
# (title_token_ids, type_id, day_of_week_id, domain_id, hour_of_day, karma, descendants, score, seq_length)
class FusionDataset(Dataset):
    def __init__(self, parquet_path, word2idx):
        self.df = pd.read_parquet(parquet_path)
        self.word2idx = word2idx
        # Print max values for debugging
        print("Max type_id:", self.df['type_id'].max())
        print("Max day_of_week_id:", self.df['day_of_week_id'].max())
        print("Max domain_id:", self.df['domain_id'].max())
        print("Num unique type_id:", self.df['type_id'].nunique())
        print("Num unique day_of_week_id:", self.df['day_of_week_id'].nunique())
        print("Num unique domain_id:", self.df['domain_id'].nunique())
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        title = row['title']
        tokens = simple_tokenize(title)
        # Convert tokens to ids (using 0 if not found)
        token_ids = [self.word2idx.get(token, self.word2idx.get('<UNK>', 0)) for token in tokens]
        seq_length = len(token_ids)
        # Extra features
        type_id = int(row['type_id'])
        day_id = int(row['day_of_week_id'])
        domain_id = int(row['domain_id'])
        # Add assertions to catch out-of-bounds indices
        assert type_id >= 0, f"type_id {type_id} < 0"
        assert day_id >= 0, f"day_id {day_id} < 0"
        assert domain_id >= 0, f"domain_id {domain_id} < 0"
        hour = float(row['hour_of_day'])
        karma = float(row['karma'])
        descendants = float(row['descendants'])
        score = float(row['score'])
        return (torch.tensor(token_ids, dtype=torch.long),
                type_id,
                day_id,
                domain_id,
                hour,
                karma,
                descendants,
                score,
                seq_length)

# Collate function to pad title token sequences
def fusion_collate_fn(batch):
    token_lists, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths = zip(*batch)
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=0)
    type_ids = torch.tensor(type_ids, dtype=torch.long)
    day_ids = torch.tensor(day_ids, dtype=torch.long)
    domain_ids = torch.tensor(domain_ids, dtype=torch.long)
    hours = torch.tensor(hours, dtype=torch.float)
    karmas = torch.tensor(karmas, dtype=torch.float)
    descendants = torch.tensor(descendants, dtype=torch.float)
    scores = torch.tensor(scores, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.float)
    return padded_tokens, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths

# Fusion Model: Uses pre-trained word2vec embeddings plus extra features for regression
class FusionModel(nn.Module):
    def __init__(self, word2vec_weight, num_words, 
                 num_types, num_days, num_domains):
        super(FusionModel, self).__init__()
        # Word2vec embedding layer; load pre-trained weights
        self.w2v_emb = nn.Embedding(num_words, EMBEDDING_DIM)
        self.w2v_emb.weight.data.copy_(word2vec_weight)
        # Optionally freeze word2vec embeddings
        self.w2v_emb.weight.requires_grad = False

        # Embedding layers for categorical features
        self.type_emb = nn.Embedding(num_types, TYPE_EMB_DIM)
        self.day_emb = nn.Embedding(num_days, DAY_EMB_DIM)
        self.domain_emb = nn.Embedding(num_domains, DOMAIN_EMB_DIM)
        
        # Fully connected layers for fusion
        fused_dim = EMBEDDING_DIM + TYPE_EMB_DIM + DAY_EMB_DIM + DOMAIN_EMB_DIM + 3  # 3 continuous features
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, 1)
        )
        
    def forward(self, title_tokens, lengths, type_ids, day_ids, domain_ids, hours, karmas, descendants):
        # title_tokens: (batch, seq_len)
        # Get word2vec embeddings for each token in the title
        with torch.no_grad():  # Don't compute gradients for frozen embeddings
            emb_tokens = self.w2v_emb(title_tokens)  # (B, L, EMBEDDING_DIM)
        # Mask out padding tokens (assumed to be 0)
        mask = (title_tokens != 0).unsqueeze(-1).float()  # (B, L, 1)
        # Sum embeddings for non-padding tokens
        summed = torch.sum(emb_tokens * mask, dim=1)  # (B, EMBEDDING_DIM)
        # Average by non-padding token count (lengths)
        lengths = lengths.unsqueeze(1).clamp(min=1)
        avg_title_emb = summed / lengths  # (B, EMBEDDING_DIM)
        # avg_title_emb is the fixed-size title vector
        
        # Categorical features embeddings
        type_feature = self.type_emb(type_ids)
        day_feature = self.day_emb(day_ids)
        domain_feature = self.domain_emb(domain_ids)
        
        # Continuous features - ensure proper device placement
        cont_features = torch.stack([hours, karmas, descendants], dim=1)
        
        # Fuse all features
        fused = torch.cat([avg_title_emb, type_feature, day_feature, domain_feature, cont_features], dim=1)
        output = self.fc(fused)
        return output.squeeze(1)

def load_word2vec_checkpoint(path, embedding_type='sgns'):
    checkpoint = torch.load(path, map_location="cpu")
    # Use specified embedding type (sgns or cbow)
    if embedding_type == 'cbow':
        weight = checkpoint['cbow_emb']
        print("Using CBOW embeddings")
    else:
        weight = checkpoint['sgns_emb']
        print("Using SGNS embeddings")
    word2idx = checkpoint['word2idx']
    return weight, word2idx

def train_fusion(model, dataloader, device, optimizer, criterion, wandb_run=None):
    model.train()
    running_loss = 0.0
    total_batches = 0
    total_samples = 0
    
    # Enable cudnn benchmarking for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Add progress tracking
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device more efficiently
        title_tokens, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths = batch
        batch_size = title_tokens.size(0)
        total_samples += batch_size
        
        title_tokens = title_tokens.to(device, non_blocking=True)
        type_ids = type_ids.to(device, non_blocking=True)
        day_ids = day_ids.to(device, non_blocking=True)
        domain_ids = domain_ids.to(device, non_blocking=True)
        hours = hours.to(device, non_blocking=True)
        karmas = karmas.to(device, non_blocking=True)
        descendants = descendants.to(device, non_blocking=True)
        scores = scores.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(title_tokens, lengths, type_ids, day_ids, domain_ids, hours, karmas, descendants)
        loss = criterion(outputs, scores)
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_loss = loss.item()
        running_loss += batch_loss * batch_size
        total_batches += 1
        
        # Update progress bar with current loss
        progress_bar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'Avg Loss': f'{running_loss/total_samples:.4f}'
        })
        
        # Log every 100 batches for more frequent monitoring
        if wandb_run is not None and batch_idx % 100 == 0:
            wandb_run.log({
                "batch_loss": batch_loss,
                "avg_loss": running_loss / total_samples,
                "batch_idx": batch_idx
            })
            
        # Print sample predictions occasionally for debugging
        if batch_idx % 1000 == 0:
            with torch.no_grad():
                sample_preds = outputs[:5].cpu().numpy()
                sample_targets = scores[:5].cpu().numpy()
                print(f"\nBatch {batch_idx} - Sample predictions vs targets:")
                for i in range(min(5, len(sample_preds))):
                    print(f"  Pred: {sample_preds[i]:.2f}, Target: {sample_targets[i]:.2f}")
    
    return running_loss / total_samples

def print_eda(parquet_path):
    df = pd.read_parquet(parquet_path)
    print("=== EDA for", parquet_path, "===")
    print("Num rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("Sample rows:\n", df.head())
    print("Describe:\n", df.describe(include='all'))
    print("Type value counts:\n", df['type_id'].value_counts())
    print("Day of week value counts:\n", df['day_of_week_id'].value_counts())
    print("Domain value counts:\n", df['domain_id'].value_counts())
    print("Score stats:\n", df['score'].describe())
    print("Karma stats:\n", df['karma'].describe())
    print("Descendants stats:\n", df['descendants'].describe())
    print("Hour of day stats:\n", df['hour_of_day'].describe())
    print("Title length stats:\n", df['title'].apply(lambda x: len(simple_tokenize(x))).describe())

def evaluate_fusion(model, dataloader, device, criterion, ablate_karma=False, ablate_descendants=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    # Enable cudnn benchmarking for better performance
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to device more efficiently
            title_tokens, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths = batch
            title_tokens = title_tokens.to(device, non_blocking=True)
            type_ids = type_ids.to(device, non_blocking=True)
            day_ids = day_ids.to(device, non_blocking=True)
            domain_ids = domain_ids.to(device, non_blocking=True)
            hours = hours.to(device, non_blocking=True)
            karmas = karmas.to(device, non_blocking=True)
            descendants = descendants.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            
            # Ablation: zero out karma and/or descendants if flag enabled
            if ablate_karma:
                karmas = torch.zeros_like(karmas)
            if ablate_descendants:
                descendants = torch.zeros_like(descendants)
            outputs = model(title_tokens, lengths, type_ids, day_ids, domain_ids, hours, karmas, descendants)
            loss = criterion(outputs, scores)
            running_loss += loss.item() * title_tokens.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(scores.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mape = (np.abs((targets - preds) / np.clip(targets, 1e-8, None))).mean()
    medae = median_absolute_error(targets, preds)
    evs = explained_variance_score(targets, preds)
    maxerr = max_error(targets, preds)
    try:
        pearson = pearsonr(targets, preds)[0]
    except Exception:
        pearson = float('nan')
    try:
        spearman = spearmanr(targets, preds)[0]
    except Exception:
        spearman = float('nan')
    metrics = {
        "loss": running_loss / len(dataloader.dataset),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "medae": medae,
        "explained_variance": evs,
        "max_error": maxerr,
        "pearson": pearson,
        "spearman": spearman,
    }
    return metrics

def print_metrics(metrics, prefix=""):
    for k, v in metrics.items():
        print(f"{prefix}{k}: {v:.4f}")

def print_system_info():
    print("=== System Info ===")
    # CPU info
    try:
        import platform
        import psutil
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
    parser = argparse.ArgumentParser(description="Fusion model to predict HN upvotes from title text + extra features")
    parser.add_argument("--train", action="store_true", help="Train the fusion model")
    parser.add_argument("--test", action="store_true", help="Evaluate the fusion model on the test set")
    parser.add_argument("--eda", action="store_true", help="Print EDA for train/test datasets")
    # New ablation flags for evaluation
    parser.add_argument("--ablate-karma", action="store_true", help="Evaluation: ablate karma feature")
    parser.add_argument("--ablate-descendants", action="store_true", help="Evaluation: ablate descendants feature")
    # Embedding type selection
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument("--sgns", action="store_true", default=True, help="Use SGNS embeddings (default)")
    embedding_group.add_argument("--cbow", action="store_true", help="Use CBOW embeddings")
    args = parser.parse_args()
    
    # Determine embedding type
    embedding_type = 'cbow' if args.cbow else 'sgns'
    
    # Print help if no arguments are provided
    if not (args.train or args.test or args.eda or args.ablate_karma or args.ablate_descendants):
        parser.print_help()
        sys.exit(0)
    
    # EDA
    if args.eda:
        print_eda(TRAIN_FILE)
        print_eda(TEST_FILE)
        sys.exit(0)

    # Exit if no command args provided
    if not args.train and not args.test:
        print("No action specified. Use --train and/or --test.")
        sys.exit(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print_system_info()
    
    # Load word2vec checkpoint with specified embedding type
    print("Loading word2vec checkpoint from:", WORD2VEC_PATH)
    w2v_weight, word2idx = load_word2vec_checkpoint(WORD2VEC_PATH, embedding_type)
    num_words = w2v_weight.size(0)
    
    # For categorical embeddings, set number of unique tokens based on processed data
    # Here we assume that type_id, day_of_week_id, domain_id are 0-indexed.
    num_types = int(os.getenv("NUM_TYPES", "10"))
    num_days = int(os.getenv("NUM_DAYS", "7"))
    num_domains = int(os.getenv("NUM_DOMAINS", "4096"))
    
    model = FusionModel(w2v_weight, num_words, num_types, num_days, num_domains)
    model.to(device)
    
    # Print model info to verify it's set up correctly
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Check if we have enough trainable parameters
    if trainable_params == 0:
        print("WARNING: No trainable parameters found!")
        sys.exit(1)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Print optimizer info
    print(f"Optimizer learning rate: {LEARNING_RATE}")
    print(f"Optimizer parameter groups: {len(optimizer.param_groups)}")
    
    # Initialize wandb
    wandb_project = os.getenv("WANDB_PROJECT", "mlx8-fusion")
    wandb_run_name = os.getenv("WANDB_RUN_NAME", "fusion")
    now_str = datetime.now().strftime("_%Y%m%d%H%M")
    run_name = wandb_run_name + f"_{embedding_type}" + now_str
    wandb.init(project=wandb_project, name=run_name, config={
        "embedding_dim": EMBEDDING_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": FUSION_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "type_emb_dim": TYPE_EMB_DIM,
        "day_emb_dim": DAY_EMB_DIM,
        "domain_emb_dim": DOMAIN_EMB_DIM,
        "fc_hidden_size": FC_HIDDEN_SIZE,
        "num_types": num_types,
        "num_days": num_days,
        "num_domains": num_domains,
        "embedding_type": embedding_type
    })

    if args.train:
        print("Loading training dataset from:", TRAIN_FILE)
        train_dataset = FusionDataset(TRAIN_FILE, word2idx)
        # Add num_workers and pin_memory for better GPU utilization
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                 collate_fn=fusion_collate_fn, num_workers=4, 
                                 pin_memory=True if torch.cuda.is_available() else False)
        
        print("Starting training for {} epochs...".format(FUSION_EPOCHS))
        for epoch in range(1, FUSION_EPOCHS+1):
            train_loss = train_fusion(model, train_loader, device, optimizer, criterion, wandb_run=wandb)
            print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}")
            wandb.log({"epoch": epoch, "train_loss": train_loss})
        
        os.makedirs(os.path.dirname(FUSION_MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), FUSION_MODEL_SAVE_PATH)
        print("Fusion model saved to", FUSION_MODEL_SAVE_PATH)
    
    if args.test:
        print("Loading test dataset from:", TEST_FILE)
        test_dataset = FusionDataset(TEST_FILE, word2idx)
        # Add num_workers and pin_memory for better GPU utilization
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                collate_fn=fusion_collate_fn, num_workers=4,
                                pin_memory=True if torch.cuda.is_available() else False)
        
        # Load saved model weights if available
        if os.path.exists(FUSION_MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(FUSION_MODEL_SAVE_PATH, map_location=device))
            print("Loaded fusion model checkpoint.")
        else:
            print("Fusion model checkpoint not found at", FUSION_MODEL_SAVE_PATH)
            sys.exit(1)
        # Pass ablation flags to evaluation function
        metrics = evaluate_fusion(model, test_loader, device, criterion, 
                                  ablate_karma=args.ablate_karma, 
                                  ablate_descendants=args.ablate_descendants)
        print_metrics(metrics, prefix="Test ")
        wandb.log({f"test_{k}": v for k, v in metrics.items()})

    wandb.finish()

if __name__ == "__main__":
    import numpy as np  # Needed for MAPE
    main()
