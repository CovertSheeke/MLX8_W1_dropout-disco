# MLX8_W1_dropout-disco

A simple PostgreSQL stack with vector support, pgAdmin, and CloudBeaver for database management.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/)

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/CovertSheeke/MLX8_W1_dropout-disco.git
   cd MLX8_W1_dropout-disco
   cd postgresql
   ```

2. **Start the services:**
   ```sh
   docker compose up -d
   ```

3. **Access the services:**
   - **PostgreSQL:**  
     Host: `localhost`  
     Port: `5432`  
     User: `example`  
     Password: `example`
   - **pgAdmin:**  
     URL: [http://localhost:5050](http://localhost:5050)  
     Email: `admin@admin.com`  
     Password: `admin`
   - **CloudBeaver:**  
     URL: [http://localhost:8978](http://localhost:8978)

4. **Stop the services:**
   ```sh
   docker compose down
   ```

## Data Persistence

Data is stored in Docker volumes and will persist between restarts.

## Notes

- The PostgreSQL image includes [pgvector](https://github.com/pgvector/pgvector) extension for vector search.
- You can manage your databases using either pgAdmin or CloudBeaver.

---

## Exploratory Data Analysis (EDA) SQL Scripts

Below are example SQL queries for EDA on the `hacker_news.items` and `hacker_news.users` tables, focused on **upvote prediction** analysis.

### a. Look for Cues (Upvote Prediction Features)

```sql
-- Count total records and score availability
SELECT 
  COUNT(*) AS total_items,
  COUNT(score) AS items_with_scores,
  COUNT(*) - COUNT(score) AS items_without_scores
FROM hacker_news.items;

-- Top 10 authors by average upvotes
SELECT "by", 
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes,
  MAX(score) AS max_upvotes
FROM hacker_news.items
WHERE score IS NOT NULL
GROUP BY "by"
HAVING COUNT(*) >= 10
ORDER BY avg_upvotes DESC
LIMIT 10;

-- Users with highest karma vs their average upvotes
SELECT u.id, u.karma, 
  AVG(i.score) AS avg_upvotes,
  COUNT(i.id) AS post_count
FROM hacker_news.users u
JOIN hacker_news.items i ON u.id = i."by"
WHERE i.score IS NOT NULL AND u.karma IS NOT NULL
GROUP BY u.id, u.karma
ORDER BY u.karma DESC
LIMIT 20;

-- Upvote patterns by posting time (hour of day)
SELECT
  EXTRACT(HOUR FROM "time") AS hour_of_day,
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes,
  MAX(score) AS max_upvotes
FROM hacker_news.items
WHERE "time" IS NOT NULL AND score IS NOT NULL
GROUP BY hour_of_day
ORDER BY avg_upvotes DESC;

-- Upvote patterns by day of week
SELECT
  EXTRACT(DOW FROM "time") AS day_of_week,
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes
FROM hacker_news.items
WHERE "time" IS NOT NULL AND score IS NOT NULL
GROUP BY day_of_week
ORDER BY avg_upvotes DESC;

-- Top performing domains by average upvotes
SELECT
  regexp_replace(url, '^https?://(?:www\.)?([^/]+).*', '\1') AS domain,
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes,
  MAX(score) AS max_upvotes
FROM hacker_news.items
WHERE url IS NOT NULL AND score IS NOT NULL
GROUP BY domain
HAVING COUNT(*) >= 50
ORDER BY avg_upvotes DESC
LIMIT 20;

-- Post type performance for upvotes
SELECT type, 
  COUNT(*) AS count,
  AVG(score) AS avg_upvotes,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) AS median_upvotes,
  MAX(score) AS max_upvotes
FROM hacker_news.items
WHERE score IS NOT NULL
GROUP BY type
ORDER BY avg_upvotes DESC;

-- Title length vs upvotes correlation
SELECT
  width_bucket(LENGTH(title), 0, 200, 10) AS title_length_bin,
  COUNT(*) AS count,
  AVG(score) AS avg_upvotes
FROM hacker_news.items
WHERE title IS NOT NULL AND score IS NOT NULL
GROUP BY title_length_bin
ORDER BY title_length_bin;
```

### b. Understand the Distributions (Upvote-Focused)

```sql
-- Upvote distribution (detailed bins)
SELECT
  CASE 
    WHEN score = 0 THEN '0'
    WHEN score = 1 THEN '1'
    WHEN score = 2 THEN '2'
    WHEN score = 3 THEN '3'
    WHEN score = 4 THEN '4'
    WHEN score = 5 THEN '5'
    WHEN score BETWEEN 6 AND 10 THEN '6-10'
    WHEN score BETWEEN 11 AND 25 THEN '11-25'
    WHEN score BETWEEN 26 AND 50 THEN '26-50'
    WHEN score BETWEEN 51 AND 100 THEN '51-100'
    WHEN score BETWEEN 101 AND 500 THEN '101-500'
    WHEN score > 500 THEN '500+'
  END AS upvote_range,
  COUNT(*) AS count,
  COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS percentage
FROM hacker_news.items
WHERE score IS NOT NULL
GROUP BY upvote_range
ORDER BY MIN(score);

-- High upvote posts (potential viral content)
SELECT COUNT(*) AS viral_posts
FROM hacker_news.items
WHERE score >= 100;

-- Zero upvote posts analysis
SELECT 
  COUNT(*) AS zero_upvote_posts,
  COUNT(*) * 100.0 / (SELECT COUNT(*) FROM hacker_news.items WHERE score IS NOT NULL) AS percentage
FROM hacker_news.items
WHERE score = 0;

-- Author karma vs post performance correlation
SELECT
  width_bucket(u.karma, 0, 10000, 10) AS karma_bin,
  COUNT(i.id) AS post_count,
  AVG(i.score) AS avg_upvotes,
  COUNT(*) FILTER (WHERE i.score >= 10) AS posts_with_10plus_upvotes
FROM hacker_news.users u
JOIN hacker_news.items i ON u.id = i."by"
WHERE u.karma IS NOT NULL AND i.score IS NOT NULL
GROUP BY karma_bin
ORDER BY karma_bin;

-- Comments vs upvotes relationship
SELECT
  width_bucket(descendants, 0, 100, 10) AS comment_bin,
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes
FROM hacker_news.items
WHERE descendants IS NOT NULL AND score IS NOT NULL
GROUP BY comment_bin
ORDER BY comment_bin;

-- Upvote statistics by year (trend analysis)
SELECT
  EXTRACT(YEAR FROM "time") AS year,
  COUNT(*) AS post_count,
  AVG(score) AS avg_upvotes,
  PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY score) AS p90_upvotes
FROM hacker_news.items
WHERE "time" IS NOT NULL AND score IS NOT NULL
GROUP BY year
ORDER BY year;
```

### c. Data Quality Issues (Upvote Prediction Focused)

```sql
-- Null value counts for key columns
SELECT
  COUNT(*) FILTER (WHERE title IS NULL) AS null_titles,
  COUNT(*) FILTER (WHERE text IS NULL) AS null_texts,
  COUNT(*) FILTER (WHERE url IS NULL) AS null_urls
FROM hacker_news.items;

-- Comprehensive null analysis for items table
SELECT
  COUNT(*) AS total_records,
  COUNT(*) FILTER (WHERE "by" IS NULL) AS null_authors,
  COUNT(*) FILTER (WHERE "time" IS NULL) AS null_timestamps,
  COUNT(*) FILTER (WHERE score IS NULL) AS null_scores,
  COUNT(*) FILTER (WHERE type IS NULL) AS null_types,
  COUNT(*) FILTER (WHERE dead IS NULL) AS null_dead_flags,
  COUNT(*) FILTER (WHERE descendants IS NULL) AS null_descendants
FROM hacker_news.items;

-- Null analysis for users table
SELECT
  COUNT(*) AS total_users,
  COUNT(*) FILTER (WHERE created IS NULL) AS null_created,
  COUNT(*) FILTER (WHERE karma IS NULL) AS null_karma,
  COUNT(*) FILTER (WHERE about IS NULL) AS null_about,
  COUNT(*) FILTER (WHERE submitted IS NULL) AS null_submitted
FROM hacker_news.users;

-- Identify potential data quality issues
SELECT
  COUNT(*) FILTER (WHERE score < 0) AS negative_scores,
  COUNT(*) FILTER (WHERE descendants < 0) AS negative_descendants,
  COUNT(*) FILTER (WHERE LENGTH(title) > 300) AS overly_long_titles,
  COUNT(*) FILTER (WHERE "time" > NOW()) AS future_timestamps
FROM hacker_news.items;

-- Check negative karma users
SELECT COUNT(*) AS negative_karma_users
FROM hacker_news.users
WHERE karma < 0;

-- Orphaned records (items without corresponding users)
SELECT COUNT(*) AS orphaned_items
FROM hacker_news.items i
LEFT JOIN hacker_news.users u ON i."by" = u.id
WHERE i."by" IS NOT NULL AND u.id IS NULL;

-- Duplicate detection
SELECT id, COUNT(*) AS duplicate_count
FROM hacker_news.items
GROUP BY id
HAVING COUNT(*) > 1;

-- Check for suspicious patterns in text content
SELECT
  COUNT(*) FILTER (WHERE text ~ '^https?://') AS text_starts_with_url,
  COUNT(*) FILTER (WHERE text ~ '[^\x00-\x7F]') AS text_with_non_ascii,
  COUNT(*) FILTER (WHERE LENGTH(text) = 0) AS empty_text_strings,
  COUNT(*) FILTER (WHERE text IS NOT NULL AND LENGTH(TRIM(text)) = 0) AS whitespace_only_text
FROM hacker_news.items
WHERE text IS NOT NULL;

-- Check for unrealistic scores (potential outliers)
SELECT
  COUNT(*) FILTER (WHERE score > 1000) AS very_high_scores,
  COUNT(*) FILTER (WHERE score = 0) AS zero_scores,
  COUNT(*) FILTER (WHERE descendants > score AND descendants > 10) AS more_comments_than_score
FROM hacker_news.items
WHERE score IS NOT NULL;

-- Posts without scores (missing target variable)
SELECT
  COUNT(*) AS posts_without_scores,
  COUNT(*) * 100.0 / (SELECT COUNT(*) FROM hacker_news.items) AS percentage_missing_scores
FROM hacker_news.items
WHERE score IS NULL;

-- Posts with scores but missing key features
SELECT
  COUNT(*) FILTER (WHERE title IS NULL AND score IS NOT NULL) AS scored_posts_no_title,
  COUNT(*) FILTER (WHERE "by" IS NULL AND score IS NOT NULL) AS scored_posts_no_author,
  COUNT(*) FILTER (WHERE "time" IS NULL AND score IS NOT NULL) AS scored_posts_no_time,
  COUNT(*) FILTER (WHERE type IS NULL AND score IS NOT NULL) AS scored_posts_no_type
FROM hacker_news.items;

-- Anomalous upvote patterns
SELECT
  COUNT(*) FILTER (WHERE score < 0) AS negative_scores,
  COUNT(*) FILTER (WHERE score > 5000) AS extremely_high_scores,
  COUNT(*) FILTER (WHERE score = 0 AND descendants > 50) AS zero_score_many_comments
FROM hacker_news.items
WHERE score IS NOT NULL;

-- Authors with posts but no user record (feature completeness)
SELECT COUNT(DISTINCT i."by") AS authors_missing_user_data
FROM hacker_news.items i
LEFT JOIN hacker_news.users u ON i."by" = u.id
WHERE i."by" IS NOT NULL AND u.id IS NULL AND i.score IS NOT NULL;

-- Data completeness for machine learning features
SELECT
  COUNT(*) AS total_posts_with_scores,
  COUNT(*) FILTER (WHERE title IS NOT NULL AND "by" IS NOT NULL AND "time" IS NOT NULL) AS complete_basic_features,
  COUNT(*) FILTER (WHERE title IS NOT NULL AND "by" IS NOT NULL AND "time" IS NOT NULL AND type IS NOT NULL) AS complete_extended_features
FROM hacker_news.items
WHERE score IS NOT NULL;
```

---

This is the repository for MLX8 week 1.

In this repo we will be attempting to create a upvote prediction model.


### HN News PostgreSQL DML
```sql
-- Table: hacker_news.items

-- DROP TABLE IF EXISTS hacker_news.items;

CREATE TABLE IF NOT EXISTS hacker_news.items
(
    id integer NOT NULL,
    dead boolean,
    type character varying(20) COLLATE pg_catalog."default",
    by character varying(255) COLLATE pg_catalog."default",
    "time" timestamp without time zone,
    text text COLLATE pg_catalog."default",
    parent integer,
    kids integer[],
    url character varying(255) COLLATE pg_catalog."default",
    score integer,
    title character varying(255) COLLATE pg_catalog."default",
    descendants integer,
    CONSTRAINT items_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS hacker_news.items
    OWNER to zer4bab;

REVOKE ALL ON TABLE hacker_news.items FROM sy91dhb;

GRANT SELECT ON TABLE hacker_news.items TO sy91dhb;

GRANT ALL ON TABLE hacker_news.items TO zer4bab;
-- Index: idx_items_by

-- DROP INDEX IF EXISTS hacker_news.idx_items_by;

CREATE INDEX IF NOT EXISTS idx_items_by
    ON hacker_news.items USING btree
    (by COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_items_time

-- DROP INDEX IF EXISTS hacker_news.idx_items_time;

CREATE INDEX IF NOT EXISTS idx_items_time
    ON hacker_news.items USING btree
    ("time" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_items_type

-- DROP INDEX IF EXISTS hacker_news.idx_items_type;

CREATE INDEX IF NOT EXISTS idx_items_type
    ON hacker_news.items USING btree
    (type COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

```


```sql
-- Table: hacker_news.users

-- DROP TABLE IF EXISTS hacker_news.users;

CREATE TABLE IF NOT EXISTS hacker_news.users
(
    id character varying(255) COLLATE pg_catalog."default" NOT NULL,
    created timestamp without time zone,
    karma integer,
    about text COLLATE pg_catalog."default",
    submitted integer[],
    CONSTRAINT users_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS hacker_news.users
    OWNER to zer4bab;

REVOKE ALL ON TABLE hacker_news.users FROM sy91dhb;

GRANT SELECT ON TABLE hacker_news.users TO sy91dhb;

GRANT ALL ON TABLE hacker_news.users TO zer4bab;
```

