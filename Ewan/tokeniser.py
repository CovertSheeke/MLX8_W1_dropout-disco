from sqlalchemy import create_engine
import pandas as pd
from collections import Counter
import time
start_time = time.time()

engine = create_engine("postgresql+psycopg2://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
connection = engine.connect()

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

def tokenize_titles(titles): 
    # Preprocess and tokenize
    all_words = []
    
    for title in titles:
        # Convert to lowercase
        title = title.lower()
        
        # Replace all punctuation with angle bracket words        
        for punct, replacement in punctuation_map.items():
            title = title.replace(punct, f' {replacement} ')
        
        # Split into words (handles multiple spaces)
        words = title.split()
        
        all_words.extend(words)
    
    # Build vocabulary (word to index mapping)
    word_counts = Counter(all_words)
    # Remove words with frequency below threshold (e.g., 2)
    threshold = 2
    word_counts = Counter({word: count for word, count in word_counts.items() if count >= threshold})
    vocab = {word: idx for idx, word in enumerate(word_counts.keys())}
    
    return vocab

titles_df = pd.read_sql_query("SELECT title FROM hacker_news.items WHERE title IS NOT NULL ORDER BY id LIMIT 100", connection)

# Usage:
result = tokenize_titles(titles_df['title'])
# print(result)
print(len(result))

print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

# Close the connection
connection.close()