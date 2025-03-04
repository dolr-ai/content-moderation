# Performance Testing Module for Content Moderation System

This module provides functionality to test the performance of the moderation server, including latency and throughput analysis.

## Features

- Sequential and concurrent request testing
- Concurrency scaling analysis
- Latency distribution analysis
- Throughput measurement
- Comprehensive performance reports with visualizations
- Support for custom test data in JSONL format

## Usage

### Command Line Interface

The performance testing module can be used through the main CLI:

```bash
python src/entrypoint.py performance --input-jsonl <path_to_test_data.jsonl>
```

#### Basic Options

- `--input-jsonl`: Path to JSONL file with texts to test (required)
- `--server-url`: URL of the moderation server (default: http://localhost:8000)
- `--output-dir`: Directory to save test results (default: performance_results)
- `--num-examples`: Number of similar examples to use in moderation (default: 3)
- `--test-type`: Type of test to run - "sequential" or "concurrent" (default: sequential)
- `--num-samples`: Number of samples to test (default: all samples in the input file)

#### Concurrent Testing Options

- `--concurrency`: Number of concurrent requests for concurrent test (default: 10)

#### Scaling Test Options

- `--run-scaling-test`: Run concurrency scaling test with multiple concurrency levels
- `--concurrency-levels`: Comma-separated list of concurrency levels for scaling test (default: 1,2,4,8,16,32)

### Example Commands

#### Run a basic sequential test

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_texts.jsonl \
    --output-dir results/performance
```

#### Run a concurrent test with 20 concurrent requests

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_texts.jsonl \
    --test-type concurrent \
    --concurrency 20
```

#### Run a concurrency scaling test

```bash
python src/entrypoint.py performance \
    --input-jsonl data/test_texts.jsonl \
    --run-scaling-test \
    --concurrency-levels 1,2,4,8,16,32,64
```

## Input Data Format

The input JSONL file should contain one JSON object per line, with at least a `text` field:

```json
{"text": "This is a test sentence for moderation."}
{"text": "Another test sentence.", "moderation_category": "clean"}
```

If a `moderation_category` field is present, it will be included in the results for comparison.

## Output

The performance test generates several outputs:

1. **JSON Results**: Detailed test results in JSON format
2. **Scaling Report**: Analysis of throughput and latency at different concurrency levels
3. **Visualizations**: Plots of latency distribution and scaling behavior
4. **Markdown Report**: Comprehensive performance report with analysis and recommendations

All outputs are saved to the specified output directory.

## Interpreting Results

The performance report includes:

- **Throughput**: Number of requests processed per second
- **Latency Statistics**: Average, median, 95th percentile, and 99th percentile latency
- **Latency Distribution**: Histogram and CDF of request latencies
- **Scaling Analysis**: How throughput and latency change with concurrency
- **System Recommendations**: Optimal concurrency level and estimated capacity

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- requests
- tqdm