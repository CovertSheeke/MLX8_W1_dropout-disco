from collections import Counter


UNK_TOKEN = "<UNK>"
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


# TODO: can we improve? e.g. remove stop words, stem/lemmatise, do frequency subsampling etc.
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
) -> dict[str, int]:
    """
    Builds a vocabulary of words that appear more than the frequency threshold.
    """
    word_counts = Counter(tokens)
    # Remove words with frequency below threshold
    word_counts = [UNK_TOKEN] + [
        word for word, count in word_counts.items() if count >= min_freq
    ]
    vocab = {word: idx for idx, word in enumerate(word_counts)}
    return vocab


def get_tokens_as_indices(tokens: list[str], vocab: dict) -> list[int]:
    """
    Converts a list of tokens to their corresponding indices using the provided vocab mapping.
    This is to ensure we have fast, random-access, constant-sized, GPU-friendly data upfront.
    """
    unk = vocab[UNK_TOKEN]
    return [vocab.get(t, unk) for t in tokens]
