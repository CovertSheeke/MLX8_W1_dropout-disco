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
from sklearn.metrics import r2_score, mean_absolute_error

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../postgresql/.data/.env"))

# Paths from environment variables
TRAIN_FILE = os.getenv("PROCESSED_TRAIN_FILE", "../postgresql/.data/hn_posts_train_processed.parquet")
TEST_FILE = os.getenv("PROCESSED_TEST_FILE", "../postgresql/.data/hn_posts_test_processed.parquet")
WORD2VEC_PATH = os.getenv("WORD2VEC_PATH", "../text8/.data/text8_compare.pt")
FUSION_MODEL_SAVE_PATH = os.getenv("FUSION_MODEL_SAVE_PATH", "../postgresql/.data/fusion_model.pt")

# Hyperparameters (example values; see .env.example for full list)
BATCH_SIZE = int(os.getenv("FUSION_BATCH_SIZE", "64"))
FUSION_EPOCHS = int(os.getenv("FUSION_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("FUSION_LEARNING_RATE", "0.001"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "100"))  # should match word2vec dims

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
        emb_tokens = self.w2v_emb(title_tokens)  # (B, L, EMBEDDING_DIM)
        # Create mask where token id != 0 (assumed padding token)
        mask = (title_tokens != 0).unsqueeze(-1).float()  # (B, L, 1)
        summed = torch.sum(emb_tokens * mask, dim=1)  # (B, EMBEDDING_DIM)
        # Avoid division by zero
        lengths = lengths.unsqueeze(1).clamp(min=1)
        avg_title_emb = summed / lengths  # (B, EMBEDDING_DIM)
        
        # Categorical features embeddings
        type_feature = self.type_emb(type_ids)
        day_feature = self.day_emb(day_ids)
        domain_feature = self.domain_emb(domain_ids)
        
        # Continuous features
        cont_features = torch.stack([hours, karmas, descendants], dim=1)
        
        # Fuse all features
        fused = torch.cat([avg_title_emb, type_feature, day_feature, domain_feature, cont_features], dim=1)
        output = self.fc(fused)
        return output.squeeze(1)

def load_word2vec_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    # Use SGNS embedding weight and word2idx mapping
    weight = checkpoint['sgns_emb']
    word2idx = checkpoint['word2idx']
    return weight, word2idx

def train_fusion(model, dataloader, device, optimizer, criterion, wandb_run=None):
    model.train()
    running_loss = 0.0
    total_batches = 0
    for batch in tqdm(dataloader, desc="Training"):
        title_tokens, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths = [b.to(device) for b in batch[:-1]] + [batch[-1].to(device)]
        optimizer.zero_grad()
        outputs = model(title_tokens, lengths, type_ids, day_ids, domain_ids, hours, karmas, descendants)
        loss = criterion(outputs, scores)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * title_tokens.size(0)
        total_batches += 1
        if wandb_run is not None:
            wandb_run.log({
                "batch_loss": loss.item()
            })
    return running_loss / len(dataloader.dataset)

def evaluate_fusion(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            title_tokens, type_ids, day_ids, domain_ids, hours, karmas, descendants, scores, lengths = [b.to(device) for b in batch[:-1]] + [batch[-1].to(device)]
            outputs = model(title_tokens, lengths, type_ids, day_ids, domain_ids, hours, karmas, descendants)
            loss = criterion(outputs, scores)
            running_loss += loss.item() * title_tokens.size(0)
            all_preds.append(outputs.cpu())
            all_targets.append(scores.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets, preds)
    return running_loss / len(dataloader.dataset), r2, mae

def main():
    parser = argparse.ArgumentParser(description="Fusion model to predict HN upvotes from title text + extra features")
    parser.add_argument("--train", action="store_true", help="Train the fusion model")
    parser.add_argument("--test", action="store_true", help="Evaluate the fusion model on the test set")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load word2vec checkpoint
    print("Loading word2vec checkpoint from:", WORD2VEC_PATH)
    w2v_weight, word2idx = load_word2vec_checkpoint(WORD2VEC_PATH)
    num_words = w2v_weight.size(0)
    
    # For categorical embeddings, set number of unique tokens based on processed data
    # Here we assume that type_id, day_of_week_id, domain_id are 0-indexed.
    num_types = int(os.getenv("NUM_TYPES", "10"))
    num_days = int(os.getenv("NUM_DAYS", "7"))
    num_domains = int(os.getenv("NUM_DOMAINS", "1000"))
    
    model = FusionModel(w2v_weight, num_words, num_types, num_days, num_domains)
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # Initialize wandb
    wandb_project = os.getenv("WANDB_PROJECT", "mlx8-fusion")
    wandb_run_name = os.getenv("WANDB_RUN_NAME", "fusion")
    now_str = datetime.now().strftime("_%Y%m%d%H%M")
    run_name = wandb_run_name + now_str
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
        "num_domains": num_domains
    })

    if args.train:
        print("Loading training dataset from:", TRAIN_FILE)
        train_dataset = FusionDataset(TRAIN_FILE, word2idx)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=fusion_collate_fn)
        
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
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=fusion_collate_fn)
        
        # Load saved model weights if available
        if os.path.exists(FUSION_MODEL_SAVE_PATH):
            model.load_state_dict(torch.load(FUSION_MODEL_SAVE_PATH, map_location=device))
            print("Loaded fusion model checkpoint.")
        else:
            print("Fusion model checkpoint not found at", FUSION_MODEL_SAVE_PATH)
            sys.exit(1)
        test_loss, test_r2, test_mae = evaluate_fusion(model, test_loader, device, criterion)
        print(f"Test Loss: {test_loss:.4f} | R2: {test_r2:.4f} | MAE: {test_mae:.4f}")
        wandb.log({"test_loss": test_loss, "test_r2": test_r2, "test_mae": test_mae})

    wandb.finish()

if __name__ == "__main__":
    main()
