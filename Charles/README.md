# Fusion Model for Hacker News Upvote Prediction

This project implements a neural network that predicts Hacker News post upvotes by fusing title text (using pre-trained Word2Vec embeddings) with categorical and continuous metadata features.

## Features Used

- **Title**: Tokenized and embedded using pre-trained Word2Vec vectors (averaged per post).
- **Type**: Post type (categorical, embedded).
- **Day of Week**: Day the post was made (categorical, embedded).
- **Domain**: Domain of the post's URL (categorical, embedded).
- **Hour of Day**: Hour the post was made (continuous).
- **Karma**: Author's karma (continuous).
- **Descendants**: Number of comments (continuous).

## Model Architecture

- **Word2Vec Embedding**: Loads pre-trained weights, frozen during training. Output size: `EMBEDDING_DIM` (default: 100).
- **Categorical Embeddings**:
  - Type: `TYPE_EMB_DIM` (default: 8)
  - Day: `DAY_EMB_DIM` (default: 3)
  - Domain: `DOMAIN_EMB_DIM` (default: 8)
- **Continuous Features**: Hour, karma, descendants (3 values).
- **Fusion**: All features are concatenated into a single vector of size  
  `EMBEDDING_DIM + TYPE_EMB_DIM + DAY_EMB_DIM + DOMAIN_EMB_DIM + 3` (default: 122).
- **Fully Connected Layers**:
  - Linear: `fused_dim` → `FC_HIDDEN_SIZE` (default: 64)
  - ReLU
  - Linear: `FC_HIDDEN_SIZE` → 1 (regression output)

## Data Preparation

- Input data is expected as parquet files with processed columns:
  - `title`, `type_id`, `day_of_week_id`, `domain_id`, `hour_of_day`, `karma`, `descendants`, `score`
- Tokenization is simple: lowercase, remove punctuation, split on spaces.
- Word2Vec checkpoint must contain `sgns_emb` (weights) and `word2idx` (vocab mapping).

## Usage

### Environment Variables

Set these in `.env` or via your environment:

- `PROCESSED_TRAIN_FILE` - Path to processed training parquet
- `PROCESSED_TEST_FILE` - Path to processed test parquet
- `WORD2VEC_PATH` - Path to pre-trained Word2Vec checkpoint
- `FUSION_MODEL_SAVE_PATH` - Where to save/load the model
- `EMBEDDING_DIM`, `TYPE_EMB_DIM`, `DAY_EMB_DIM`, `DOMAIN_EMB_DIM`, `FC_HIDDEN_SIZE`, `FUSION_BATCH_SIZE`, `FUSION_EPOCHS`, `FUSION_LEARNING_RATE`, `NUM_TYPES`, `NUM_DAYS`, `NUM_DOMAINS`

### Training

```sh
python Charles/feature-fusion.py --train
```

### Testing

```sh
python Charles/feature-fusion.py --test
```

## File Structure

- `Charles/feature-fusion.py`: Main model, dataset, training, and evaluation code.

## Model Input Sizes (default)

| Feature         | Size |
|-----------------|------|
| Title (Word2Vec)| 100  |
| Type            | 8    |
| Day of Week     | 3    |
| Domain          | 8    |
| Hour            | 1    |
| Karma           | 1    |
| Descendants     | 1    |
| **Total**       | 122  |

## References

- [PyTorch Documentation](https://pytorch.org/)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)

---