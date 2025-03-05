"""
Command-line interface for content moderation system
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional

# Import from parent package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import config
from src.servers.sglang_servers import ServerManager
from src.vectors.vector_db import VectorDB
from src.moderation.moderation_system import ModerationSystem
from src.performance.performance_tester import run_performance_test

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def add_server_parser(subparsers):
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


def add_vectordb_parser(subparsers):
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


def add_moderation_parser(subparsers):
    """Add content moderation-related arguments to the parser"""
    moderation_parser = subparsers.add_parser("moderate", help="Moderate content")
    moderation_parser.add_argument("--text", type=str, help="Text to moderate")
    moderation_parser.add_argument(
        "--file", type=str, help="JSONL file with texts to moderate"
    )
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


def add_moderation_server_parser(subparsers):
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
    return mod_server_parser


def add_performance_parser(subparsers):
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
    return perf_parser


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Content Moderation System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add parsers for different commands
    add_server_parser(subparsers)
    add_vectordb_parser(subparsers)
    add_moderation_parser(subparsers)
    add_moderation_server_parser(subparsers)
    add_performance_parser(subparsers)

    return parser.parse_args()


def run_server_command(args):
    """Run server command"""
    # Initialize server manager
    server_manager = ServerManager(
        hf_token=config.hf_token,
        llm_model=args.llm_model,
        llm_port=args.llm_port,
        emb_model=args.emb_model,
        emb_port=args.emb_port,
        mem_fraction_llm=args.mem_fraction_llm,
        mem_fraction_emb=args.mem_fraction_emb,
        max_requests=args.max_requests,
    )

    # Determine which servers to start
    start_llm = args.llm
    start_embedding = args.embedding

    if not (start_llm or start_embedding):
        logger.error("No servers specified. Use --llm and/or --embedding.")
        return False

    # Run servers
    return server_manager.run_servers(
        start_embedding=start_embedding, start_llm=start_llm
    )


def run_vectordb_command(args):
    """Run vector database command"""
    if not args.create:
        logger.error("No vector database command specified. Use --create.")
        return False

    if not args.input_jsonl:
        logger.error("No input JSONL file specified. Use --input-jsonl.")
        return False

    if not args.save_dir:
        logger.error("No save directory specified. Use --save-dir.")
        return False

    # Initialize vector database manager
    vector_db = VectorDB(
        base_url="http://localhost:8890/v1",  # Use default embedding server
        api_key="None",
    )

    # Create vector database
    try:
        # Analyze the JSONL file structure first
        import json

        text_field = "text"  # Default field

        try:
            with open(args.input_jsonl, "r") as f:
                # Read the first line to understand structure
                first_line = f.readline().strip()
                if first_line:
                    data = json.loads(first_line)
                    if isinstance(data, dict):
                        logger.info(f"JSONL data structure: {list(data.keys())}")
                        if "text" in data:
                            text_field = "text"
                        else:
                            # Use the first field if text is not available
                            text_field = list(data.keys())[0]
                        logger.info(f"Using field '{text_field}' for text content")
        except Exception as e:
            logger.warning(f"Could not analyze JSONL structure: {e}")
            logger.warning(f"Using default field '{text_field}'")

        index, metadata_df, embeddings = vector_db.create_vector_db_from_jsonl(
            input_jsonl=args.input_jsonl,
            save_dir=args.save_dir,
            text_field=text_field,
            batch_size=args.batch_size,
            sample_size=args.sample,
            prune_text_to_max_chars=args.prune_text_to_max_chars,
        )
        logger.info(f"Created vector database with {index.ntotal} vectors")
        return True
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        logger.error(
            "Please check that your JSONL file contains text strings in the 'text' field"
        )
        logger.error(
            'Example format: {"text": "your text content", "moderation_category": "clean"}'
        )
        return False


def run_moderation_command(args):
    """Run moderation command"""
    if not args.db_path:
        logger.error("No vector database path specified. Use --db-path.")
        return False

    if not (args.text or args.file):
        logger.error("No text or file specified. Use --text or --file.")
        return False

    # Initialize moderation system
    system = ModerationSystem(
        embedding_url="http://localhost:8890/v1",
        llm_url="http://localhost:8899/v1",
        vector_db_path=args.db_path,
        prompt_path=args.prompt_path,
    )

    try:
        # Moderate single text
        if args.text:
            result = system.classify_text(args.text, num_examples=args.num_examples)
            print(json.dumps(result, indent=2))
            return True

        # Moderate texts from file
        if args.file:
            texts = []
            with open(args.file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    texts.append(data.get("text", ""))

            results = system.batch_classify(texts, num_examples=args.num_examples)

            # Save results to output file if specified
            if args.output:
                with open(args.output, "w") as f:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                logger.info(f"Saved moderation results to {args.output}")
            else:
                print(json.dumps(results, indent=2))

            return True
    except Exception as e:
        logger.error(f"Error moderating content: {e}")
        return False


def run_moderation_server_command(args):
    """Run moderation server command"""
    from src.servers.moderation_server import run_server

    return run_server(
        vector_db_path=args.db_path,
        prompt_path=args.prompt_path,
        host=args.host,
        port=args.port,
        embedding_url=args.embedding_url,
        llm_url=args.llm_url,
    )


def run_performance_command(args):
    """Run performance testing command"""
    # Parse concurrency levels if provided
    concurrency_levels = None
    if args.concurrency_levels:
        try:
            concurrency_levels = [
                int(level) for level in args.concurrency_levels.split(",")
            ]
        except ValueError:
            logger.error(
                "Invalid concurrency levels format. Use comma-separated integers."
            )
            return False

    # Run performance test
    try:
        results = run_performance_test(
            input_file=args.input_jsonl,
            server_url=args.server_url,
            output_dir=args.output_dir,
            num_examples=args.num_examples,
            test_type=args.test_type,
            num_samples=args.num_samples,
            concurrency=args.concurrency,
            run_scaling_test=args.run_scaling_test,
            concurrency_levels=concurrency_levels,
        )

        if "error" in results:
            logger.error(f"Performance test failed: {results['error']}")
            return False

        # Print summary information
        if "summary" in results:
            summary = results["summary"]
            logger.info("Performance Test Summary:")
            logger.info(f"- Total samples: {summary['total_samples']}")
            logger.info(f"- Total time: {summary['total_time_seconds']:.2f} seconds")
            logger.info(
                f"- Throughput: {summary['throughput_requests_per_second']:.2f} requests/second"
            )
            logger.info(
                f"- Average latency: {summary['avg_latency_seconds']*1000:.2f} ms"
            )
            logger.info(
                f"- 95th percentile latency: {summary['p95_latency_seconds']*1000:.2f} ms"
            )

        # Print report path
        if "report_path" in results:
            logger.info(f"Performance report saved to: {results['report_path']}")

        return True
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        return False


def main():
    """Main function"""
    args = parse_args()

    if args.command == "server":
        success = run_server_command(args)
    elif args.command == "vectordb":
        success = run_vectordb_command(args)
    elif args.command == "moderate":
        success = run_moderation_command(args)
    elif args.command == "moderation-server":
        success = run_moderation_server_command(args)
    elif args.command == "performance":
        success = run_performance_command(args)
    else:
        print(
            "No command specified. Use server, vectordb, moderate, moderation-server, or performance."
        )
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
