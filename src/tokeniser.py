from collections import Counter
import random
import logging

logger = logging.getLogger(__name__)

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
DOC_START_TOKEN = "<DOC_START>"
DOC_END_TOKEN = "<DOC_END>"
PUNCTUATION_MAP = {
    "<": "<LESS>",
    ">": "<GREATER>",
    ",": "<COMMA>",
    ".": "<PERIOD>",
    "!": "<EXCLAMATION>",
    "?": "<QUESTION>",
    ":": "<COLON>",
    ";": "<SEMICOLON>",
    "-": "<DASH>",
    "(": "<LPAREN>",
    ")": "<RPAREN>",
    "[": "<LBRACKET>",
    "]": "<RBRACKET>",
    "{": "<LBRACE>",
    "}": "<RBRACE>",
    '"': "<QUOTE>",
    "'": "<APOSTROPHE>",
    "/": "<SLASH>",
    "\\": "<BACKSLASH>",
    "&": "<AMPERSAND>",
    "@": "<AT>",
    "#": "<HASH>",
    "$": "<DOLLAR>",
    "%": "<PERCENT>",
    "*": "<ASTERISK>",
    "+": "<PLUS>",
    "=": "<EQUALS>",
    "|": "<PIPE>",
    "~": "<TILDE>",
    "`": "<BACKTICK>",
}


# TODO: can we improve? e.g. remove stop words, stem/lemmatise
def tokenise(text: str) -> list[str]:
    """
    Tokenises a long string of text by lowercasing, replacing punctuation with predefined angle bracket words.

    Args:
        text (str): A single string.

    Returns:
        dict: A dictionary mapping each word to a unique index.
    """
    # Convert to lowercase
    text = text.lower()

    # Replace all punctuation with angle bracket words
    for punct, replacement in PUNCTUATION_MAP.items():
        text = text.replace(punct, f" {replacement} ")

    # Split into words (handles multiple spaces)
    words = text.split()
    return words


def build_vocab(
    tokens: list[str],
    min_freq: int = 5,
    subsampling_threshold: float = 1e-5,
) -> dict[str, int]:
    """
    Builds a vocabulary of words that appear more than the frequency threshold.
    """
    word_counts = Counter(tokens)
    # Remove words with frequency below threshold
    token_list = [UNK_TOKEN, PAD_TOKEN, DOC_START_TOKEN, DOC_END_TOKEN] + [
        word for word, count in word_counts.items() if count >= min_freq
    ]
    num_discarded_freq = len(word_counts) - len(token_list)

    # Frequency subsampling (remove frequent words with probability proportional to their frequency)
    if subsampling_threshold is not None:
        logger.info(f"Subsampling tokens with threshold: {subsampling_threshold}")
        total_count = sum(Counter(tokens).values())
        subsampled = []
        freqs = Counter(tokens)
        discarded_count_subsampling = 0  # Counter for discarded tokens
        for word in token_list:
            if word == UNK_TOKEN:
                subsampled.append(word)
                continue
            freq = freqs[word] / total_count
            prob_discard = 1 - (subsampling_threshold / freq) ** 0.5
            if random.random() > prob_discard:
                subsampled.append(word)
            else:
                discarded_count_subsampling += 1  # Increment counter if discarded
        token_list = subsampled

    vocab = {word: idx for idx, word in enumerate(token_list)}

    # Report
    logger.info(f"Total tokens in: {len(tokens)}")
    logger.info(
        f"Number discarded from frequency threshold: {num_discarded_freq} ({num_discarded_freq / len(word_counts) * 100:.2f}%)"
    )
    if subsampling_threshold is not None:
        logger.info(
            f"Number discarded from subsampling: {discarded_count_subsampling} ({discarded_count_subsampling / len(token_list) * 100:.2f}%)"
        )
    logger.info(f"Vocab size: {len(vocab)}")
    return vocab


def get_tokens_as_indices(tokens: list[str], vocab: dict) -> list[int]:
    """
    Converts a list of tokens to their corresponding indices using the provided vocab mapping.
    This is to ensure we have fast, random-access, constant-sized, GPU-friendly data upfront.
    """
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]


def get_words_from_indices(indeces: list[int], vocab: dict) -> list[str]:
    """
    Converts a list of token indices to a list of token values
    """
    return [
        list(vocab.keys())[list(vocab.values()).index(idx)]
        for idx in indeces
        if idx in vocab.values()
    ]
