

from word2vec.data import get_text8
from word2vec.tokeniser import build_vocab, get_tokens_as_indices, tokenise


def generate_skipgram_pairs(corpus, context_size):
    pairs = []
    for i in range(context_size, len(corpus) - context_size):
        center = corpus[i]
        context = corpus[i - context_size:i] + corpus[i + 1:i + context_size + 1]
        for ctx in context:
            pairs.append((center, ctx))
    return pairs


def build_sgram_dataset(context_size: int = 5, txt_8_path: str = "data/text8.txt") -> tuple[list[tuple[int, int]], dict]:
    # read text8 file
    # Read the text8 file
    with open(txt8_path, "r", encoding="utf-8") as f:
        text = f.read()

    text_tokens = tokenise(text)

    # Build the vocabulary
    vocab = build_vocab(text_tokens, min_freq=0, subsampling_threshold=1e-4)
    
    print(text_tokens[:10])  # Print first 10 tokens for verification

    text_token_inds = get_tokens_as_indices(text_tokens, vocab)

    print(text_token_inds[:10])  # Print first 10 token indices for verification
    # print(len(text_token_inds))

    # Generate skip-gram pairs
    skipgram_pairs = generate_skipgram_pairs(text_token_inds, context_size)
    print(skipgram_pairs[:10])  # Print first 10 pairs for verification
    
    # print(len(skipgram_pairs))  # Print total number of pairs
    return skipgram_pairs, vocab

txt8_path = get_text8()