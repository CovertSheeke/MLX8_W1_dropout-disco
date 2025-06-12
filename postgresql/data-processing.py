# data_processing.py
import re
import json
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from urllib.parse import urlparse # For domain extraction
import os
import sys
from tqdm import tqdm

TRAIN_FILE = os.environ.get("TRAIN_FILE", "./.data/hn_posts_train.parquet")
TEST_FILE = os.environ.get("TEST_FILE", "./.data/hn_posts_test.parquet")
PROCESSED_TRAIN_FILE = os.environ.get("PROCESSED_TRAIN_FILE", "./.data/hn_posts_train_processed.parquet")
PROCESSED_TEST_FILE = os.environ.get("PROCESSED_TEST_FILE", "./.data/hn_posts_test_processed.parquet")
VOCAB_PATH = os.environ.get("VOCAB_PATH", "./.data/vocabulary.json")
DOMAIN_PARQUET_PATH = os.environ.get("DOMAIN_PARQUET_PATH", "./.data/domain.parquet")
MIN_DOMAIN_FREQ = int(os.environ.get("MIN_DOMAIN_FREQ", 100))
MAX_TITLE_TOKENS = int(os.environ.get("MAX_TITLE_TOKENS", 50))

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

    NOTE: If using a pre-trained word2vec model, you do NOT need to build your own vocabulary.
    The vocabulary from the pre-trained model should be used for token-to-vector mapping.
    This function is left here for reference and for cases where you want to train embeddings from scratch.
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
    """
    Converts a list of tokens to a list of token IDs.

    NOTE: If using a pre-trained word2vec model, you do NOT need to convert tokens to integer IDs using a custom vocabulary.
    Instead, you should look up the word vectors directly from the pre-trained model.
    This function is left here for reference and for cases where you want to train embeddings from scratch.
    """
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
    def __init__(self, df, vocabulary=None, max_len=None):
        self.titles = df['title'].tolist()
        self.scores = df['score'].tolist()
        self.max_len = max_len # Optional: pad/truncate titles to a max length

        # Pre-process additional features and store them as lists
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

        # Tokenization and vocabulary are not used anymore.
        # Instead, you should use your pre-trained word2vec tokenizer/vectorizer in your model pipeline.
        # Here, just return the raw title string.
        # If you want to pad/truncate, do it in your collate_fn or model pipeline.

        # For backward compatibility, return an empty tensor for token_ids.
        token_ids = torch.empty(0, dtype=torch.long)

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
            token_ids,  # placeholder, not used
            features,
            torch.tensor(score, dtype=torch.float)
        )

def process_domains(train_path, test_path, domain_parquet_path):
    """
    Processes both train and test datasets to build a domain-to-id mapping,
    saves the mapping as a parquet file.

    Only loads the 'url' column to save memory.
    Displays progress and domain counts for train and test.
    """
    print("Processing domains from train and test datasets...")
    # Only load 'url' column to save memory
    df_train = pd.read_parquet(train_path, columns=['url'])
    df_test = pd.read_parquet(test_path, columns=['url'])

    print(f"Train set: {len(df_train)} rows, Test set: {len(df_test)} rows")

    # Merge before extraction for memory efficiency and progress bar
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total records to process: {len(df_all)}")

    # Extract domains with tqdm progress bar
    df_all['domain'] = [extract_domain(url) for url in tqdm(df_all['url'], desc="Extracting domains")]

    # Show top domains in each split
    train_domain_counts = Counter(df_all.loc[:len(df_train)-1, 'domain'].dropna())
    test_domain_counts = Counter(df_all.loc[len(df_train):, 'domain'].dropna())
    print("\nTop 10 domains in train set:")
    for domain, count in train_domain_counts.most_common(10):
        print(f"  {domain}: {count}")
    print("\nTop 10 domains in test set:")
    for domain, count in test_domain_counts.most_common(10):
        print(f"  {domain}: {count}")

    # Count all domains for mapping
    domain_counts = Counter(df_all['domain'].dropna())
    min_domain_freq = MIN_DOMAIN_FREQ  # Use value from .env
    frequent_domains = {d for d, c in domain_counts.items() if c >= min_domain_freq}
    print(f"\nNumber of domains with >= {min_domain_freq} occurrences: {len(frequent_domains)}")

    domain_vocab = {'<UNK_DOMAIN>': 0}
    for domain in sorted(list(frequent_domains)):
        if domain not in domain_vocab:
            domain_vocab[domain] = len(domain_vocab)
    # Save as parquet
    domain_df = pd.DataFrame(list(domain_vocab.items()), columns=['domain', 'domain_id'])
    print(f"Saving Domain mapping to {domain_parquet_path} ...")

    domain_df.to_parquet(domain_parquet_path, index=False)
    
def load_domain_vocab_from_parquet(domain_parquet_path):
    """
    Loads domain-to-id mapping from a parquet file and returns as a dict.
    """
    domain_df = pd.read_parquet(domain_parquet_path)
    return dict(zip(domain_df['domain'], domain_df['domain_id']))

# --- 6. Feature Engineering Functions ---
def apply_feature_engineering(df, split):
    """
    Applies feature engineering to the DataFrame in-place.
    Handles type, time, domain features.
    split: "train" or "test"
    """
    # Process 'type'
    if 'unknown' not in TYPE_MAPPING:
        TYPE_MAPPING['unknown'] = max(TYPE_MAPPING.values()) + 1
    print("Mapping types to type_id...")
    df['type_id'] = [TYPE_MAPPING.get(t, TYPE_MAPPING['unknown']) for t in tqdm(df['type'], desc="Type mapping")]

    # Process 'time'
    print("Parsing time column...")
    df['time'] = pd.to_datetime(df['time'])
    print("Extracting hour_of_day...")
    df['hour_of_day'] = [t.hour for t in tqdm(df['time'], desc="Hour of day")]
    print("Extracting day_of_week...")
    df['day_of_week'] = [t.dayofweek for t in tqdm(df['time'], desc="Day of week")]
    print("Mapping day_of_week to day_of_week_id...")
    df['day_of_week_id'] = [DAY_OF_WEEK_MAPPING.get(d, 0) for d in tqdm(df['day_of_week'], desc="Day of week mapping")]

    # Process 'url' for 'domain'
    print("Extracting domains from URLs...")
    df['domain'] = [extract_domain(url) for url in tqdm(df['url'], desc="Extracting domains")]

    # --- Domain Vocabulary ---
    global DOMAIN_VOCAB, DOMAIN_UNK_ID
    if os.path.exists(DOMAIN_PARQUET_PATH):
        DOMAIN_VOCAB = load_domain_vocab_from_parquet(DOMAIN_PARQUET_PATH)
        DOMAIN_UNK_ID = DOMAIN_VOCAB['<UNK_DOMAIN>']
    else:
        print(f"Domain parquet file not found at {DOMAIN_PARQUET_PATH}. Please run with --domain first.")
        sys.exit(1)

    print("Mapping domains to IDs...")
    df['domain_id'] = [get_domain_id(domain) for domain in tqdm(df['domain'], desc="Mapping domain to ID")]
    df['domain_id'] = pd.Series(df['domain_id'], dtype=int)
    print(f"Number of unique domains (after freq filter): {len(DOMAIN_VOCAB)}")
    print(f"Sample Domain IDs: {df['domain_id'].value_counts().head()}")

    print("\nSample processed data with new features:")
    print(df[['title', 'score', 'type_id', 'hour_of_day', 'day_of_week_id', 'karma', 'descendants', 'domain_id']].head())
    return None

# --- Main execution example for data_processing.py ---
if __name__ == "__main__":
    
    # Handle --domain with overwrite prompt
    if "--domain" in sys.argv:
        if os.path.exists(DOMAIN_PARQUET_PATH):
            resp = input(f"{DOMAIN_PARQUET_PATH} already exists. Overwrite? [Y/n]: ").strip().lower()
            if resp not in ("y", "yes", ""):
                print("Aborted domain processing.")
                sys.exit(0)
        process_domains(TRAIN_FILE, TEST_FILE, DOMAIN_PARQUET_PATH)
        sys.exit(0)
    
    if not os.path.exists(DOMAIN_PARQUET_PATH):
        print(f"Domain parquet file not found at {DOMAIN_PARQUET_PATH}.")
        print("Please run: python data-processing.py --domain")
        sys.exit(1)

    # Handle --domain-list [N]
    if "--domain-list" in sys.argv:
        idx = sys.argv.index("--domain-list")
        try:
            n = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            n = 100
        if not os.path.exists(DOMAIN_PARQUET_PATH):
            print(f"Domain parquet file not found at {DOMAIN_PARQUET_PATH}.")
            print("Please run: python data-processing.py --domain")
            sys.exit(1)
        domain_df = pd.read_parquet(DOMAIN_PARQUET_PATH)
        # Show top domains by domain_id, but sort by most frequent (highest domain_id first)
        print(f"Top {n} domains in {DOMAIN_PARQUET_PATH}:")
        print(domain_df.sort_values("domain_id", ascending=False).head(n).to_string(index=False))
        sys.exit(0)

    split = os.environ.get("SPLIT", "train")
    if "--test" in sys.argv:
        print("Overriding split to 'test' due to command line argument.")
        split = "test"
    elif "--train" in sys.argv:
        print("Overriding split to 'train' due to command line argument.")
        split = "train"

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
    apply_feature_engineering(df_hn, split)

    # --- Save processed data ---
    df_hn.to_parquet(processed_path, index=False)
    print(f"\nProcessed {split} data saved to {processed_path}")

    # Show DataFrame info and describe for the processed split
    print(f"\nProcessed {split} DataFrame info:")
    df_hn.info()
    print(f"\nProcessed {split} DataFrame describe:")
    print(df_hn.describe(include='all'))

    print("\nAll data processing completed.")