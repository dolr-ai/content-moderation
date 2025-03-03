#!/usr/bin/env python3
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

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Content Moderation System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server commands
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

    # Vector DB commands
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

    # Moderation commands
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
        "--examples", type=int, default=3, help="Number of similar examples to use"
    )

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
    )

    try:
        # Moderate single text
        if args.text:
            result = system.classify_text(args.text, num_examples=args.examples)
            print(json.dumps(result, indent=2))
            return True

        # Moderate texts from file
        if args.file:
            texts = []
            with open(args.file, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    texts.append(data.get("text", ""))

            results = system.batch_classify(texts, num_examples=args.examples)

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


def main():
    """Main function"""
    args = parse_args()

    if args.command == "server":
        success = run_server_command(args)
    elif args.command == "vectordb":
        success = run_vectordb_command(args)
    elif args.command == "moderate":
        success = run_moderation_command(args)
    else:
        print("No command specified. Use server, vectordb, or moderate.")
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
