from sqlalchemy import create_engine
import pandas as pd
from tokeniser import tokenise
import time
start_time = time.time()

engine = create_engine("postgresql+psycopg2://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
connection = engine.connect()


titles_df = pd.read_sql_query("SELECT title FROM hacker_news.items WHERE title IS NOT NULL ORDER BY id LIMIT 100000", connection)
# Concatenate all titles into one long string
text = " ".join(titles_df['title'].tolist())

# Usage:
result = tokenise(text, frequency_threshold=10)
print(result)
print(len(result))

print("Elapsed time: {:.2f} seconds".format(time.time() - start_time))

# Close the connection
connection.close()