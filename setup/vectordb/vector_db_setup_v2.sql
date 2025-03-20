-- ========================================================================
-- OPTIMIZED VECTOR SEARCH SCHEMA FOR BIGQUERY COMMENT MODERATION
-- ========================================================================
-- This schema implements best practices for high-performance vector search
-- using BigQuery including partitioning, clustering, and optimized indexes

-- ========================================================================
-- STEP 1: Create optimized base table with partitioning and clustering
-- ========================================================================
-- Partitioning by date enables efficient time-based filtering
-- Clustering by moderation_category improves performance for category filters
CREATE OR REPLACE TABLE stage_test_tables.test_comment_mod_embeddings (
  text STRING,                 -- The original text content
  moderation_category STRING,  -- Classification category (e.g., "clean", "toxic", etc)
  embedding ARRAY<FLOAT64>,    -- Vector embedding for similarity search (typically 768 or 1536 dimensions)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),  -- Timestamp for partitioning
  source STRING  -- source of data
)
PARTITION BY DATE(created_at)
CLUSTER BY moderation_category;

-- ========================================================================
-- STEP 2: Insert sample data (reference only)
-- ========================================================================
-- Note: Sample data should be inserted here
-- Refer to vector_db_dummy_rows.sql for example data

-- ========================================================================
-- STEP 3: Create optimized vector index for ANN search
-- ========================================================================
-- IVF (Inverted File) provides efficient approximate nearest neighbor search
-- for vector embeddings at scale
CREATE OR REPLACE VECTOR INDEX moderation_vector_index
ON stage_test_tables.test_comment_mod_embeddings(embedding)
OPTIONS(
  distance_type = 'COSINE',
  index_type = 'IVF',
  ivf_options = '{"num_lists": 150}'
);

-- ========================================================================
-- STEP 4: Create combined search index for text and category search
-- ========================================================================
-- Enables efficient full-text search across multiple columns
CREATE SEARCH INDEX combined_search_index
ON stage_test_tables.test_comment_mod_embeddings(text, moderation_category);

-- ========================================================================
-- STEP 5: Note about additional filtering optimization
-- ========================================================================
-- Note: BigQuery doesn't support traditional indexes like CREATE INDEX
-- Instead, it relies on partitioning and clustering for query optimization
-- The table is already optimized with:
--   1. Partitioning by DATE(created_at)
--   2. Clustering by moderation_category
-- These should provide good performance for most filtering operations

-- ========================================================================
-- STEP 6: Verify index creation
-- ========================================================================
-- Query information schema to confirm vector index was created successfully
SELECT * FROM stage_test_tables.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE table_name = 'test_comment_mod_embeddings';

-- Verify search index was created successfully
SELECT * FROM stage_test_tables.INFORMATION_SCHEMA.SEARCH_INDEXES
WHERE table_name = 'test_comment_mod_embeddings';

-- Note: No additional indexes to verify beyond vector and search indexes

-- ========================================================================
-- STEP 7: Example optimized vector search queries
-- ========================================================================

-- Example 1: Vector search using a randomly selected embedding
WITH random_embedding AS (
  SELECT embedding
  FROM stage_test_tables.test_comment_mod_embeddings
  WHERE embedding IS NOT NULL  -- Ensure we get a valid embedding
  ORDER BY RAND()
  LIMIT 1
)

SELECT
  base.text,
  base.moderation_category,
  distance
FROM
  VECTOR_SEARCH(
    (
      SELECT * FROM stage_test_tables.test_comment_mod_embeddings
    ),
    'embedding',
    (SELECT embedding FROM random_embedding),
    top_k => 5,
    distance_type => 'COSINE',
    options => '{"fraction_lists_to_search": 0.1}'
  )
ORDER BY distance
LIMIT 5;



-- ========================================================================
-- STEP 8: Maintenance tasks (run periodically)
-- ========================================================================

-- Analyze table statistics for better query planning
-- Run this when data distribution changes significantly
ANALYZE TABLE stage_test_tables.test_comment_mod_embeddings;

-- Monitor index performance
SELECT
  table_name,
  index_name,
  last_refresh_time,
  row_count,
  index_size_bytes / POW(1024, 3) AS index_size_gb
FROM
  stage_test_tables.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE
  table_name = 'test_comment_mod_embeddings';