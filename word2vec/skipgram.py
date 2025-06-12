from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import logging
import os
from data import get_text8
from tokeniser import build_vocab, get_tokens_as_indices, tokenise
import wandb

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# constants (excluding hyperparameters)
RNG_SEED = 42
CHECKPOINT_FREQUENCY = 5  # how often to save model weights during training
MODEL_NAME = "word2vec_{}_{}_{}"  # format: architecture, dataset, wandb run
DEBUG_MODE = os.environ.get("DEBUG_MODE", "0") == "1"
WANDB_TEAM = "freemvmt-london"
WANDB_PROJECT = "word2vec"

# hot swap essential config from command line
EPOCHS = int(os.environ.get("EPOCHS", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2048))



config = {
    "epochs": EPOCHS,
    "seed": 42,
    "batch_size": BATCH_SIZE,
    "architecture": "skipgram",
    "dataset": "text8",
    "embedding_dim": 100,
    "learning_rate": 0.01,
    "context_size": 1,
    "min_freq": 0,
    "subsampling_threshold": 1e-4,
    "val_split": 0.1,
    "test_split": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def generate_skipgram_pairs(corpus, context_size):
    pairs = []
    for i in range(context_size, len(corpus) - context_size):
        center = corpus[i]
        context = corpus[i - context_size:i] + corpus[i + 1:i + context_size + 1]
        for ctx in context:
            pairs.append((center, ctx))
    return pairs


def build_sgram_dataset(context_size: int = None, txt_8_path: str = "data/text8.txt") -> tuple[list[tuple[int, int]], dict]:
    if context_size is None:
        context_size = config["context_size"]
    # Read the text8 file
    with open(txt_8_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_tokens = tokenise(text)

    # Build the vocabulary
    vocab = build_vocab(
        text_tokens,
        min_freq=config["min_freq"],
        subsampling_threshold=config["subsampling_threshold"]
    )
    
    text_token_inds = get_tokens_as_indices(text_tokens, vocab)

    # Generate skip-gram pairs
    skipgram_pairs = generate_skipgram_pairs(text_token_inds, context_size)
    
    return skipgram_pairs, vocab

logger = logging.getLogger(__name__)
def train_skipgram_model(
    skipgram_pairs: list[tuple[int, int]],
):
    """
    Train a skip-gram model on the provided skipgram pairs.
    """
    
    # This is a placeholder for the actual training logic.
    # You would typically use a library like PyTorch or TensorFlow to implement the model.
    logger.info("Training skip-gram model on provided pairs...")

    # Example: Use a neural network to learn word embeddings based on skipgram pairs
    # For now, we just log the number of pairs
    logger.info(f"Number of skipgram pairs: {len(skipgram_pairs)}")
    return

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, skipgram_pairs):
        self.pairs = skipgram_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)

class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context):
        center_emb = self.in_embeddings(center)
        context_emb = self.out_embeddings(context)
        score = torch.sum(center_emb * context_emb, dim=1)
        return score

# Split skipgram_pairs into train, val, test sets (80/10/10 split)
def split_dataset(pairs, seed=None):
    logger.info("Starting split_dataset")
    if seed is None:
        seed = config["seed"]
    train_pairs, temp_pairs = train_test_split(pairs, test_size=config["val_split"] + config["test_split"], random_state=seed)
    logger.info(f"Split into train ({len(train_pairs)}) and temp ({len(temp_pairs)}) pairs")
    val_size = config["val_split"] / (config["val_split"] + config["test_split"])
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=1 - val_size, random_state=seed)
    logger.info(f"Split temp into val ({len(val_pairs)}) and test ({len(test_pairs)}) pairs")
    return train_pairs, val_pairs, test_pairs


def train_one_epoch(model, dataloader, optimizer, loss_fn, vocab_size, device):
    logger.info("Starting train_one_epoch")
    model.train()
    total_loss = 0
    for batch_idx, (center, context) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        logger.debug(f"Batch {batch_idx}: center shape {center.shape}, context shape {context.shape}")
        center, context = center.to(device), context.to(device)
        optimizer.zero_grad()
        # Positive samples
        pos_labels = torch.ones(center.size(0), device=device)
        pos_score = model(center, context)
        pos_loss = loss_fn(pos_score, pos_labels)
        # Negative sampling
        neg_context = torch.randint(0, vocab_size, context.size(), device=device)
        neg_labels = torch.zeros(center.size(0), device=device)
        neg_score = model(center, neg_context)
        neg_loss = loss_fn(neg_score, neg_labels)
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}: loss {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Finished train_one_epoch with avg_loss {avg_loss:.4f}")
    return avg_loss

def evaluate_one_epoch(model, dataloader, loss_fn, vocab_size, device):
    logger.info("Starting evaluate_one_epoch")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (center, context) in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            logger.debug(f"Eval Batch {batch_idx}: center shape {center.shape}, context shape {context.shape}")
            center, context = center.to(device), context.to(device)
            pos_labels = torch.ones(center.size(0), device=device)
            pos_score = model(center, context)
            pos_loss = loss_fn(pos_score, pos_labels)
            neg_context = torch.randint(0, vocab_size, context.size(), device=device)
            neg_labels = torch.zeros(center.size(0), device=device)
            neg_score = model(center, neg_context)
            neg_loss = loss_fn(neg_score, neg_labels)
            loss = pos_loss + neg_loss
            total_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f"Eval Batch {batch_idx}: loss {loss.item():.4f}")
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Finished evaluate_one_epoch with avg_loss {avg_loss:.4f}")
    return avg_loss

def orchestrate_training(
    train_pairs,
    val_pairs,
    test_pairs,
    vocab_size,
    embedding_dim=None,
    batch_size=None,
    lr=None,
    epochs=None,
    device=None
):
    logger.info("Starting orchestrate_training")
    if embedding_dim is None:
        embedding_dim = config["embedding_dim"]
    if batch_size is None:
        batch_size = config["batch_size"]
    if lr is None:
        lr = config["learning_rate"]
    if epochs is None:
        epochs = config["epochs"]
    if device is None:
        device = config["device"]

    logger.info(f"Training config: embedding_dim={embedding_dim}, batch_size={batch_size}, lr={lr}, epochs={epochs}, device={device}")

    # Initialize wandb
    try:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_TEAM,
            config={
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "architecture": config["architecture"],
                "dataset": config["dataset"],
            },
            reinit=True,
        )
        logger.info("wandb initialized successfully")
    except Exception as e:
        logger.error(f"wandb initialization failed: {e}")

    train_dataset = SkipGramDataset(train_pairs)
    val_dataset = SkipGramDataset(val_pairs)
    test_dataset = SkipGramDataset(test_pairs)

    logger.info(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    logger.info("DataLoaders created")

    model = SkipGramModel(vocab_size, embedding_dim).to(device)
    logger.info("Model created and moved to device")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs} starting")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, vocab_size, device)
        val_loss = evaluate_one_epoch(model, val_loader, loss_fn, vocab_size, device)
        logger.info(f"Epoch {epoch+1}: Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        try:
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        except Exception as e:
            logger.error(f"wandb.log failed: {e}")

    test_loss = evaluate_one_epoch(model, test_loader, loss_fn, vocab_size, device)
    logger.info(f"Test loss: {test_loss:.4f}")
    try:
        wandb.log({"test_loss": test_loss})
        wandb.finish()
    except Exception as e:
        logger.error(f"wandb.log or finish failed: {e}")
    logger.info("orchestrate_training finished")
    return model


txt8_path = get_text8()
logger.info(f"Using text8 dataset at: {txt8_path}")

ds_pairs, ds_vocab = build_sgram_dataset(context_size=config["context_size"], txt_8_path=txt8_path)
logger.info(f"Built skip-gram dataset with {len(ds_pairs)} pairs and vocabulary size {len(ds_vocab)}")

train_pairs, val_pairs, test_pairs = split_dataset(ds_pairs)
# Example usage
model = orchestrate_training(train_pairs, val_pairs, test_pairs, len(ds_vocab))

