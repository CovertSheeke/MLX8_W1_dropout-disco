# db_utils.py
import os
from sqlalchemy import create_engine, text
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
TOTAL_RECORDS_TO_FETCH = int(os.getenv("TOTAL_RECORDS_TO_FETCH", "200000"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TRAIN_FILE = os.getenv("TRAIN_FILE", "./data/hn_posts_train.parquet")
TEST_FILE = os.getenv("TEST_FILE", "./data/hn_posts_test.parquet")

def get_db_engine():
    """Creates and returns a SQLAlchemy engine for the Hacker News database."""
    print(f"Attempting to connect to: {DB_CONNECTION_STRING.split('@')[-1]}") # Print without credentials
    engine = create_engine(DB_CONNECTION_STRING)
    return engine

def fetch_hn_items(engine, limit=None):
    """
    Fetches Hacker News items (specifically 'story', 'poll', 'pollopt' types),
    focusing on titles and scores, with an optional limit.
    Args:
        engine: SQLAlchemy engine.
        limit (int, optional): Maximum number of records to fetch. Defaults to None (all).
    Returns:
        pd.DataFrame: DataFrame of filtered Hacker News items.
    """
    print("Preparing to fetch items from hacker_news.items ...")
    query = """
    SELECT
        id,
        title,
        score,
        "by" AS author,
        "time",
        url,
        type,
        descendants
    FROM
        hacker_news.items
    WHERE
        score IS NOT NULL
        AND title IS NOT NULL
        AND type IN ('story', 'poll', 'pollopt') -- Explicitly filter for desired types
    """
    if limit:
        query += f" LIMIT {limit}" # Add limit clause
    print("Executing SQL query for items ...")
    with engine.connect() as connection:
        df_items = pd.read_sql_query(text(query), connection)
    print(f"Fetched {len(df_items)} items of desired types.")
    return df_items

def fetch_hn_users(engine):
    """Fetches Hacker News user data, focusing on karma."""
    print("Preparing to fetch users from hacker_news.users ...")
    query = """
    SELECT
        id,
        karma
    FROM
        hacker_news.users
    WHERE
        karma IS NOT NULL;
    """
    print("Executing SQL query for users ...")
    with engine.connect() as connection:
        df_users = pd.read_sql_query(text(query), connection)
    print(f"Fetched {len(df_users)} users.")
    return df_users

if __name__ == "__main__":
    engine = get_db_engine()
    print("Starting data fetching and processing ...")
    df_items = fetch_hn_items(engine, limit=TOTAL_RECORDS_TO_FETCH)
    df_users = fetch_hn_users(engine)
    print("Merging items and users DataFrames ...")
    df_merged = pd.merge(df_items, df_users, left_on='author', right_on='id', how='left')
    df_merged.rename(columns={'id_y': 'user_id', 'id_x': 'item_id'}, inplace=True)
    df_merged['karma'].fillna(0, inplace=True)
    df_merged['descendants'].fillna(0, inplace=True)
    print(f"\nTotal merged records after type filtering: {len(df_merged)}")
    print(f"Posts with missing author karma: {df_merged['karma'].isnull().sum()}")
    df_cleaned = df_merged.dropna(subset=['score', 'title']).copy()
    print("Splitting data into train and test sets ...")
    df_train, df_test = train_test_split(
        df_cleaned,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\nTrain set size: {len(df_train)} records ({len(df_train)/len(df_cleaned):.1%})")
    print(f"Test set size: {len(df_test)} records ({len(df_test)/len(df_cleaned):.1%})")
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(TEST_FILE), exist_ok=True)
    print("Saving train and test sets to parquet files ...")
    df_train.to_parquet(TRAIN_FILE, index=False)
    df_test.to_parquet(TEST_FILE, index=False)
    print(f"\nTraining data saved to {TRAIN_FILE}")
    print(f"Test data saved to {TEST_FILE}")
    print("\nTrain DataFrame info:")
    df_train.info()
    print("\nTrain DataFrame describe:")
    print(df_train.describe(include='all'))
    print("\nTest DataFrame info:")
    df_test.info()
    print("\nTest DataFrame describe:")
    print(df_test.describe(include='all'))
    print("\nData fetching and splitting complete.")