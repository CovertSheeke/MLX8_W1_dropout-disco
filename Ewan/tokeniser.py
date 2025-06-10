import pandas as pd
from collections import Counter

punctuation_map = {
    '<': '<LESS>', '>': '<GREATER>', ',': '<COMMA>', '.': '<PERIOD>', 
    '!': '<EXCLAMATION>', '?': '<QUESTION>',
    ':': '<COLON>', ';': '<SEMICOLON>', '-': '<DASH>', '(': '<LPAREN>',
    ')': '<RPAREN>', '[': '<LBRACKET>', ']': '<RBRACKET>', '{': '<LBRACE>',
    '}': '<RBRACE>', '"': '<QUOTE>', "'": '<APOSTROPHE>', '/': '<SLASH>',
    '\\': '<BACKSLASH>', '&': '<AMPERSAND>', '@': '<AT>', '#': '<HASH>',
    '$': '<DOLLAR>', '%': '<PERCENT>', '*': '<ASTERISK>', '+': '<PLUS>',
    '=': '<EQUALS>', '|': '<PIPE>',
    '~': '<TILDE>', '`': '<BACKTICK>'
}

def tokeniser(text, frequency_threshold): 
    """
    Tokenises a long string of text by lowercasing, replacing punctuation with predefined angle bracket words,
    and building a vocabulary of words that appear more than the frequency threshold.

    Args:
        text (str): A single string.

    Returns:
        dict: A dictionary mapping each word to a unique index.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace all punctuation with angle bracket words        
    for punct, replacement in punctuation_map.items():
        text = text.replace(punct, f' {replacement} ')
    
    # Split into words (handles multiple spaces)
    words = text.split() 
    
    # Build vocabulary (word to index mapping)
    word_counts = Counter(words)
    # Remove words with frequency below threshold (e.g., 2)
    threshold = 2
    word_counts = Counter({word: count for word, count in word_counts.items() if count >= frequency_threshold})
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    
    return vocab