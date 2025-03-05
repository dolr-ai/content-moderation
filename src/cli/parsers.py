"""
Parsers for the CLI
"""

import argparse


def add_server_parser(subparsers: argparse.ArgumentParser):
    """Add server-related arguments to the parser"""
    server_parser = subparsers.add_parser("server", help="Manage SGLang servers")
    server_parser.add_argument("--llm", action="store_true", help="Start LLM server")
    server_parser.add_argument(
        "--embedding", action="store_true", help="Start embedding server"
    )
    server_parser.add_argument(
        "--llm-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="LLM model to use",
    )
    server_parser.add_argument(
        "--llm-port", type=int, default=8899, help="Port for LLM server"
    )
    server_parser.add_argument(
        "--emb-model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Embedding model to use",
    )
    server_parser.add_argument(
        "--emb-port", type=int, default=8890, help="Port for embedding server"
    )
    server_parser.add_argument(
        "--mem-fraction-llm",
        type=float,
        default=0.80,
        help="Fraction of GPU memory to use for LLM",
    )
    server_parser.add_argument(
        "--mem-fraction-emb",
        type=float,
        default=0.25,
        help="Fraction of GPU memory to use for embedding model",
    )
    server_parser.add_argument(
        "--max-requests",
        type=int,
        default=32,
        help="Maximum number of concurrent requests",
    )
    return server_parser


def add_vectordb_parser(subparsers: argparse.ArgumentParser):
    """Add vector database-related arguments to the parser"""
    vectordb_parser = subparsers.add_parser("vectordb", help="Manage vector database")
    vectordb_parser.add_argument(
        "--create", action="store_true", help="Create vector database"
    )
    vectordb_parser.add_argument(
        "--input-jsonl", type=str, help="Input JSONL file for creating vector database"
    )
    vectordb_parser.add_argument(
        "--save-dir", type=str, help="Directory to save vector database"
    )
    vectordb_parser.add_argument(
        "--sample", type=int, help="Number of records to sample from input JSONL"
    )
    vectordb_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embedding creation"
    )
    vectordb_parser.add_argument(
        "--prune-text-to-max-chars",
        default=2000,
        type=int,
        help="Prune text to max characters",
    )
    vectordb_parser.add_argument(
        "--index-type",
        type=str,
        default="IP",
        help="Index type for vector database",
        choices=["IP", "L2"],
    )
    return vectordb_parser


def add_moderation_parser(subparsers: argparse.ArgumentParser):
    """Add content moderation-related arguments to the parser"""
    moderation_parser = subparsers.add_parser("moderate", help="Moderate content")
    moderation_parser.add_argument("--text", type=str, help="Text to moderate")
    moderation_parser.add_argument(
        "--output", type=str, help="Output file for moderation results"
    )
    moderation_parser.add_argument(
        "--db-path", type=str, help="Path to vector database directory"
    )
    moderation_parser.add_argument(
        "--num-examples", type=int, default=3, help="Number of similar examples to use"
    )
    moderation_parser.add_argument(
        "--prompt-path", type=str, help="Path to prompts file"
    )
    moderation_parser.add_argument(
        "--max-input-length",
        type=int,
        default=2000,
        help="Maximum input length",
        required=True,
    )
    return moderation_parser


def add_moderation_server_parser(subparsers: argparse.ArgumentParser):
    """Add moderation server-related arguments to the parser"""
    mod_server_parser = subparsers.add_parser(
        "moderation-server", help="Run moderation API server"
    )
    mod_server_parser.add_argument(
        "--db-path", type=str, required=True, help="Path to vector database directory"
    )
    mod_server_parser.add_argument(
        "--prompt-path", type=str, required=True, help="Path to prompts file"
    )
    mod_server_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to"
    )
    mod_server_parser.add_argument(
        "--port", type=int, default=8000, help="Port for moderation server"
    )
    mod_server_parser.add_argument(
        "--max-input-length",
        type=int,
        default=2000,
        help="Maximum input length",
        required=True,
    )
    mod_server_parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8890/v1",
        help="URL for embedding server",
    )
    mod_server_parser.add_argument(
        "--llm-url",
        type=str,
        default="http://localhost:8899/v1",
        help="URL for LLM server",
    )
    mod_server_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes for the server",
    )
    return mod_server_parser


def add_performance_parser(subparsers: argparse.ArgumentParser):
    """Add performance testing-related arguments to the parser"""
    perf_parser = subparsers.add_parser(
        "performance", help="Run performance tests on moderation server"
    )
    perf_parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Input JSONL file with texts to test",
    )
    perf_parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the moderation server",
    )
    perf_parser.add_argument(
        "--output-dir",
        type=str,
        default="performance_results",
        help="Directory to save test results",
    )
    perf_parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of similar examples to use in moderation",
    )
    perf_parser.add_argument(
        "--test-type",
        type=str,
        default="sequential",
        choices=["sequential", "concurrent"],
        help="Type of test to run (sequential or concurrent)",
    )
    perf_parser.add_argument(
        "--num-samples", type=int, help="Number of samples to test (None for all)"
    )
    perf_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests for concurrent test",
    )
    perf_parser.add_argument(
        "--run-scaling-test", action="store_true", help="Run concurrency scaling test"
    )
    perf_parser.add_argument(
        "--concurrency-levels",
        type=str,
        help="Comma-separated list of concurrency levels for scaling test",
    )
    perf_parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip generating visualizations after testing",
    )
    return perf_parser


def add_perf_visualization_parser(subparsers: argparse.ArgumentParser):
    """Add visualization-related arguments to the parser"""
    viz_parser = subparsers.add_parser(
        "visualize", help="Visualize performance test results"
    )
    viz_parser.add_argument(
        "--results-file", type=str, help="Path to performance results JSON file"
    )
    viz_parser.add_argument(
        "--scaling-report-file", type=str, help="Path to scaling report JSON file"
    )
    viz_parser.add_argument(
        "--output-dir",
        type=str,
        default="performance_visualizations",
        help="Directory to save visualizations",
    )
    viz_parser.add_argument(
        "--no-report", action="store_true", help="Do not generate comprehensive report"
    )
    return viz_parser
