# Word2Vec on Text8

This folder contains an implementation of **Word2Vec** (both Skip-Gram with Negative Sampling (SGNS) and Continuous Bag of Words (CBOW)) trained on the [text8](http://mattmahoney.net/dc/text8.zip) dataset using PyTorch.

## Features

- **Download and preprocess text8**: Script downloads and extracts the text8 corpus.
- **Vocabulary building and subsampling**: Filters rare words and applies word frequency subsampling.
- **SGNS and CBOW models**: Implements both architectures with negative sampling.
- **Efficient PyTorch Datasets**: Custom datasets for SGNS and CBOW.
- **Training and logging**: Supports batch training, progress bars, and Weights & Biases (wandb) logging.
- **Evaluation utilities**: Includes functions for nearest neighbors and analogy tasks.
- **Configurable via environment variables**: Most hyperparameters can be set in a `.env` file.

## Usage

1. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

2. **Download the dataset**:
   ```
   python word2vec-text8.py --download
   ```

3. **Train the models**:
   ```
   python word2vec-text8.py --train
   ```

   Training logs and metrics will be sent to wandb if configured.

4. **Test embeddings (nearest neighbors, analogies)**:
   ```
   python word2vec-text8.py --test
   ```

   Or run tests after training with:
   ```
   python word2vec-text8.py --train --test
   ```

## Configuration

You can create a `.env` file in this directory to override defaults. Example:

```
TEXT8_URL=http://mattmahoney.net/dc/text8.zip
EMBEDDING_DIM=100
BATCH_SIZE=8192
EPOCHS=3
MIN_COUNT=5
LEARNING_RATE=0.001
WANDB_PROJECT=text8-word2vec
```

## Output

- Trained embeddings and model state are saved to `.data/text8_compare.pt`.
- Embeddings can be used for downstream NLP tasks or further analysis.

## References

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [text8 Dataset](http://mattmahoney.net/dc/text8.zip)
