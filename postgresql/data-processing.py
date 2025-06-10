# data_processing.py
import re
import json
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from urllib.parse import urlparse # For domain extraction
import os

TRAIN_FILE = os.environ.get("TRAIN_FILE", "./.data/hn_posts_train.parquet")
TEST_FILE = os.environ.get("TEST_FILE", "./.data/hn_posts_test.parquet")
PROCESSED_TRAIN_FILE = os.environ.get("PROCESSED_TRAIN_FILE", "./.data/hn_posts_train_processed.parquet")
PROCESSED_TEST_FILE = os.environ.get("PROCESSED_TEST_FILE", "./.data/hn_posts_test_processed.parquet")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "./.data/vocabulary.json")
DOMAIN_VOCAB_PATH = os.environ.get("DOMAIN_VOCAB_PATH", "./.data/domain_vocabulary.json")

# --- 1. Constants and Mappings ---
# Define a mapping for 'type' (can be learned as embeddings later or one-hot encoded)
# For now, let's map them to integers.
TYPE_MAPPING = {
    'story': 0, #5,351,748
    'comment': 1, # 35,731,362, Though we only use titles for now, comments might appear if we filter by type
    'job': 2, # 17,001
    'poll': 3, # 2,113
    'pollopt': 4, #14,721
  #  'ask', 'show', 'unknown': 5, 6, 7
}

NUM_TYPES = len(TYPE_MAPPING) # For potential embedding layer

# Define a mapping for day of week (0=Monday, 6=Sunday for PyTorch embeddings)
# EXTRACT(DOW FROM "time") gives 0 for Sunday, 1 for Monday, ..., 6 for Saturday
# We'll stick to that convention.
DAY_OF_WEEK_MAPPING = {
    0: 0, # Sunday
    1: 1, # Monday
    2: 2, # Tuesday
    3: 3, # Wednesday
    4: 4, # Thursday
    5: 5, # Friday
    6: 6  # Saturday
}
NUM_DAYS_OF_WEEK = len(DAY_OF_WEEK_MAPPING) # For potential embedding layer

# --- 2. Basic Tokenization Function (same as before) ---
def simple_tokenize(text):
    """
    Performs basic tokenization:
    - Converts to lowercase.
    - Removes punctuation (except for internal hyphens/apostrophes if desired, but simplifying for now).
    - Splits by whitespace.
    """
    if not isinstance(text, str):
        return [] # Handle non-string input, e.g., NaNs
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

# --- 3. Vocabulary Building (same as before) ---
def build_vocabulary(tokenized_texts, min_freq=5):
    """
    Builds a vocabulary mapping tokens to IDs.
    Includes special tokens: <PAD>, <UNK>.
    Removes tokens below a certain frequency.
    """
    all_tokens = [token for sublist in tokenized_texts for token in sublist]
    token_counts = Counter(all_tokens)

    filtered_tokens = {token for token, count in token_counts.items() if count >= min_freq}

    special_tokens = ['<PAD>', '<UNK>']
    vocabulary = {token: i for i, token in enumerate(special_tokens)}
    
    for token in sorted(list(filtered_tokens)):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    
    id_to_token = {v: k for k, v in vocabulary.items()}
    
    print(f"Vocabulary size: {len(vocabulary)}")
    return vocabulary, id_to_token

def tokens_to_ids(tokens, vocabulary):
    """Converts a list of tokens to a list of token IDs."""
    unk_id = vocabulary.get('<UNK>')
    return [vocabulary.get(token, unk_id) for token in tokens]

# --- 4. Feature Engineering Functions for Non-Textual Data ---
def extract_domain(url):
    """Extracts the base domain from a URL."""
    if pd.isna(url):
        return None
    try:
        domain = urlparse(url).netloc
        # Remove 'www.' prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain if domain else None # Return None if domain is empty string
    except:
        return None # Handle malformed URLs

# Global mapping for domains, populated during initial processing
DOMAIN_VOCAB = {}
DOMAIN_UNK_ID = 0 # Default ID for unknown domains
# This will be updated after processing the entire dataset
# during the initial run in __main__ block.

def get_domain_id(domain):
    """Converts a domain string to its integer ID."""
    if domain in DOMAIN_VOCAB:
        return DOMAIN_VOCAB[domain]
    return DOMAIN_UNK_ID # Return UNK ID for unseen domains


# --- 5. PyTorch Dataset for Regression (Hacker News Titles + Features) ---
class HNTitlesAndFeaturesDataset(Dataset):
    def __init__(self, df, vocabulary, max_len=None):
        self.titles = df['title'].tolist()
        self.scores = df['score'].tolist()
        self.vocabulary = vocabulary
        self.max_len = max_len # Optional: pad/truncate titles to a max length

        # Pre-process additional features and store them as lists
        # Ensure these columns exist and are appropriately type-casted before passing df
        self.types = df['type_id'].tolist()
        self.hours = df['hour_of_day'].tolist()
        self.days_of_week = df['day_of_week_id'].tolist()
        self.karmas = df['karma'].tolist()
        self.descendants = df['descendants'].tolist()
        self.domain_ids = df['domain_id'].tolist()


    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        title = self.titles[idx]
        score = self.scores[idx]

        # Tokenize and convert title to IDs
        tokens = simple_tokenize(title)
        token_ids = tokens_to_ids(tokens, self.vocabulary)

        if self.max_len:
            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                pad_id = self.vocabulary.get('<PAD>')
                token_ids = token_ids + [pad_id] * (self.max_len - len(token_ids))
        
        # Prepare non-textual features
        # Ensure these are float or long as appropriate for PyTorch.
        # Categorical features that might become embeddings: type, day_of_week, domain
        # Numerical features: hour, karma, descendants
        
        type_id = torch.tensor(self.types[idx], dtype=torch.long)
        hour_of_day = torch.tensor(self.hours[idx], dtype=torch.float)
        day_of_week_id = torch.tensor(self.days_of_week[idx], dtype=torch.long)
        karma = torch.tensor(self.karmas[idx], dtype=torch.float)
        descendants = torch.tensor(self.descendants[idx], dtype=torch.float)
        domain_id = torch.tensor(self.domain_ids[idx], dtype=torch.long)

        # Combine all non-textual features into a single tensor or dict for flexibility
        # For Early Fusion, a single concatenated tensor is often used.
        # For Late Fusion, they might be passed separately.
        # Let's create a single tensor for now, assuming early fusion.
        # Be careful with dtypes: categorical features should be long, numerical floats.
        
        # Concatenate numerical features directly. For categorical ones,
        # we might pass their IDs to an Embedding layer in the model,
        # or one-hot encode them here if the model doesn't have an embedding layer for them.
        # For now, let's pass all feature IDs (type, day, domain) and numerical features (hour, karma, descendants) separately.
        # The model will then decide how to handle them.
        
        # A dictionary might be more robust to manage different feature types
        features = {
            'type_id': type_id,
            'hour_of_day': hour_of_day,
            'day_of_week_id': day_of_week_id,
            'karma': karma,
            'descendants': descendants,
            'domain_id': domain_id
        }

        return (
            torch.tensor(token_ids, dtype=torch.long), 
            features, 
            torch.tensor(score, dtype=torch.float)
        )

# --- Main execution example for data_processing.py ---
if __name__ == "__main__":
    # Choose which split to process
    split = os.environ.get("SPLIT", "train")  # "train" or "test"
    if split == "train":
        parquet_path = TRAIN_FILE
        processed_path = PROCESSED_TRAIN_FILE
    else:
        parquet_path = TEST_FILE
        processed_path = PROCESSED_TEST_FILE

    df_hn = pd.read_parquet(parquet_path)
    print(f"Loaded {split} data from {parquet_path}")

    # Show DataFrame info 
    print("\nDataFrame info:")
    df_hn.info()
    print("\nDataFrame describe:")
    print(df_hn.describe(include='all'))

    # --- 6. Apply Feature Engineering to DataFrame ---
    print("\nApplying feature engineering...")

    # Process 'type'
    # Add 'unknown' to TYPE_MAPPING if not present
    if 'unknown' not in TYPE_MAPPING:
        TYPE_MAPPING['unknown'] = max(TYPE_MAPPING.values()) + 1
    df_hn['type_id'] = df_hn['type'].map(TYPE_MAPPING).fillna(TYPE_MAPPING['unknown']).astype(int)

    # Process 'time'
    df_hn['time'] = pd.to_datetime(df_hn['time'])
    df_hn['hour_of_day'] = df_hn['time'].dt.hour
    df_hn['day_of_week'] = df_hn['time'].dt.dayofweek # Monday=0, Sunday=6
    df_hn['day_of_week_id'] = df_hn['day_of_week'].map(DAY_OF_WEEK_MAPPING).fillna(0).astype(int)

    # Process 'url' for 'domain'
    df_hn['domain'] = df_hn['url'].apply(extract_domain)

    # --- Domain Vocabulary ---
    if split == "train":
        # Build domain vocab from train set
        domain_counts = Counter(df_hn['domain'].dropna())
        min_domain_freq = 50
        frequent_domains = {d for d, c in domain_counts.items() if c >= min_domain_freq}
        DOMAIN_VOCAB = {'<UNK_DOMAIN>': 0}
        for domain in sorted(list(frequent_domains)):
            if domain not in DOMAIN_VOCAB:
                DOMAIN_VOCAB[domain] = len(DOMAIN_VOCAB)
        DOMAIN_UNK_ID = DOMAIN_VOCAB['<UNK_DOMAIN>']
        # Save domain vocab
        with open(DOMAIN_VOCAB_PATH, "w") as f:
            json.dump(DOMAIN_VOCAB, f)
        print(f"Domain vocabulary saved to {DOMAIN_VOCAB_PATH}")
    else:
        # Load domain vocab from train
        with open(DOMAIN_VOCAB_PATH, "r") as f:
            DOMAIN_VOCAB = json.load(f)
        DOMAIN_UNK_ID = DOMAIN_VOCAB['<UNK_DOMAIN>']

    df_hn['domain_id'] = df_hn['domain'].apply(get_domain_id).astype(int)
    print(f"Number of unique domains (after freq filter): {len(DOMAIN_VOCAB)}")
    print(f"Sample Domain IDs: {df_hn['domain_id'].value_counts().head()}")

    # --- Tokenization & Vocabulary ---
    df_hn['tokenized_title'] = df_hn['title'].apply(simple_tokenize)
    if split == "train":
        vocab, id_to_token = build_vocabulary(df_hn['tokenized_title'].tolist(), min_freq=5)
        with open(VOCAB_PATH, "w") as f:
            json.dump(vocab, f)
        print(f"Title vocabulary saved to {VOCAB_PATH}")
    else:
        with open(VOCAB_PATH, "r") as f:
            vocab = json.load(f)
    df_hn['title_ids'] = df_hn['tokenized_title'].apply(lambda x: tokens_to_ids(x, vocab))

    print("\nSample processed data with new features:")
    print(df_hn[['title', 'score', 'type_id', 'hour_of_day', 'day_of_week_id', 'karma', 'descendants', 'domain_id']].head())

    # --- Save processed data ---
    df_hn.to_parquet(processed_path, index=False)
    print(f"\nProcessed {split} data saved to {processed_path}")

    # Show DataFrame info and describe for the processed split
    print(f"\nProcessed {split} DataFrame info:")
    df_hn.info()
    print(f"\nProcessed {split} DataFrame describe:")
    print(df_hn.describe(include='all'))

    # --- Create PyTorch Dataset and DataLoader ---
    # For demonstration, let's use a max_len of 50 for padding/truncation
    hn_dataset = HNTitlesAndFeaturesDataset(df_hn, vocab, max_len=50)
    hn_dataloader = DataLoader(hn_dataset, batch_size=32, shuffle=True)

    print(f"\nDataset size: {len(hn_dataset)} posts")
    print(f"Number of batches (batch_size=32): {len(hn_dataloader)}")

    # Get one batch from the DataLoader to verify output
    sample_batch_token_ids, sample_batch_features, sample_batch_scores = next(iter(hn_dataloader))
    
    print("\nSample Batch of Token IDs (first post):")
    print(sample_batch_token_ids[0])
    print(f"Shape: {sample_batch_token_ids.shape}")

    print("\nSample Batch of Features (first post):")
    for k, v in sample_batch_features.items():
        print(f"  {k}: {v[0].item()} (Shape: {v.shape}, Dtype: {v.dtype})")
    
    print("\nSample Batch of Scores (first post):")
    print(sample_batch_scores[0].item())
    print(f"Shape: {sample_batch_scores.shape}")

    print("\nAll data processing completed.")