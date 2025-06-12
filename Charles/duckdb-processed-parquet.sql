-- Need to fix: install duckdb cli
-- > duckdb
-- > (the above works)
SELECT * FROM '../postgresql/.data/hn_posts_train_processed.parquet' LIMIT 10;

-- Set path to the processed parquet file
PRAGMA disable_progress_bar;
SET parquet_file='../postgresql/.data/hn_posts_train_processed.parquet';

-- Preview the schema and a few rows
DESCRIBE SELECT * FROM read_parquet($parquet_file);
SELECT * FROM read_parquet($parquet_file) LIMIT 10;

-- Count total rows
SELECT COUNT(*) AS total_rows FROM read_parquet($parquet_file);

-- Basic statistics for numeric columns
SELECT
  COUNT(*) AS n,
  AVG(score) AS avg_score,
  MIN(score) AS min_score,
  MAX(score) AS max_score,
  STDDEV(score) AS std_score,
  AVG(karma) AS avg_karma,
  AVG(descendants) AS avg_descendants
FROM read_parquet($parquet_file);

-- Distribution of type_id
SELECT type_id, COUNT(*) AS count
FROM read_parquet($parquet_file)
GROUP BY type_id
ORDER BY count DESC;

-- Distribution of day_of_week_id
SELECT day_of_week_id, COUNT(*) AS count
FROM read_parquet($parquet_file)
GROUP BY day_of_week_id
ORDER BY day_of_week_id;

-- Top 10 domains by frequency
SELECT domain_id, COUNT(*) AS count
FROM read_parquet($parquet_file)
GROUP BY domain_id
ORDER BY count DESC
LIMIT 10;

-- Hour of day distribution
SELECT hour_of_day, COUNT(*) AS count
FROM read_parquet($parquet_file)
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- Correlation between score and karma
SELECT corr(score, karma) AS corr_score_karma FROM read_parquet($parquet_file);

-- Titles with highest scores
SELECT title, score, karma, descendants
FROM read_parquet($parquet_file)
ORDER BY score DESC
LIMIT 10;

-- Titles with lowest scores
SELECT title, score, karma, descendants
FROM read_parquet($parquet_file)
ORDER BY score ASC
LIMIT 10;

-- Average score by type_id
SELECT type_id, AVG(score) AS avg_score, COUNT(*) AS n
FROM read_parquet($parquet_file)
GROUP BY type_id
ORDER BY avg_score DESC;

-- Average score by domain_id (top 10)
SELECT domain_id, AVG(score) AS avg_score, COUNT(*) AS n
FROM read_parquet($parquet_file)
GROUP BY domain_id
ORDER BY avg_score DESC
LIMIT 10;
