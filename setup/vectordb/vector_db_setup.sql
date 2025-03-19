-- STEP 1: Create base table for moderation content and embeddings
-- This table stores the text content, moderation categories, and vector embeddings
CREATE TABLE org_dataset.comment_moderation_embeddings (
  text STRING,                 -- The original text content
  moderation_category STRING,  -- Classification category (e.g., "clean")
  embedding ARRAY<FLOAT32>     -- Vector embedding for similarity search
);

-- STEP 2: Insert sample data
-- Note: Sample data should be inserted here
-- Refer to vector_db_dummy_rows.sql for example data

-- STEP 3: Create vector index for similarity search
-- This enables efficient approximate nearest neighbor (ANN) search
CREATE VECTOR INDEX moderation_vector_index
ON org_dataset.comment_moderation_embeddings(embedding)
OPTIONS(
  distance_type = 'COSINE',      -- Use cosine similarity as distance measure
  index_type = 'IVF',            -- Inverted file index type for approximate search
  ivf_options = '{"num_lists": 150}'  -- Number of partitions for the index
);

-- STEP 4: Create combined search index
-- This enables full-text search on both text and moderation_category columns
CREATE SEARCH INDEX combined_search_index
ON org_dataset.comment_moderation_embeddings(text, moderation_category);

-- STEP 5: Verify index creation
-- Query information schema to confirm vector index was created successfully
SELECT * FROM org_dataset.INFORMATION_SCHEMA.VECTOR_INDEXES
WHERE table_name = 'comment_moderation_embeddings';

-- STEP 6: Example vector search query with metadata filtering
-- This query demonstrates how to find similar content with specific filtering
SELECT
  candidates.text,
  candidates.moderation_category,
  distance  -- Lower values indicate higher similarity
FROM
  VECTOR_SEARCH(
    TABLE filtered_candidates,    -- Use the prefiltered table
    'embedding',                  -- Column containing embeddings
    TABLE query_embedding,        -- Our query embedding
    top_k => 5,                  -- Return top 5 results
    distance_type => 'COSINE',    -- Use cosine distance
    options => '{"fraction_lists_to_search": 0.1}'  -- Search 10% of index partitions
  );

-- STEP 7: Example vector search across all categories
-- This query finds similar content without category filtering
SELECT
  base.text,
  base.moderation_category,
  distance
FROM
  VECTOR_SEARCH(
    TABLE org_dataset.comment_moderation_embeddings,  -- Search the entire table
    'embedding',
    (SELECT embedding FROM org_dataset.comment_moderation_embeddings ORDER BY RAND() LIMIT 1),  -- Use a random vector
    top_k => 5,
    distance_type => 'COSINE',
    options => '{"fraction_lists_to_search": 0.1}'  -- Search 10% of index partitions
  );

