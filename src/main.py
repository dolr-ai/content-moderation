#!/usr/bin/env python3
"""
Main entry point for the content moderation system.

This script provides a command-line interface to the various components
of the content moderation system.
"""
import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path

# Add path handling for imports
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

# Use relative or absolute imports based on how the script is being run
if __name__ == "__main__" or "src" not in __name__:
    # Running as script or from outside the package
    from src.config.config import config
    from src.server.sglang_server import main as server_main
    from src.vector_db.setup_db import main as setup_db_main
    from src.inference.moderator import main as moderator_main
else:
    # Running from within the package
    from .config.config import config
    from .server.sglang_server import main as server_main
    from .vector_db.setup_db import main as setup_db_main
    from .inference.moderator import main as moderator_main


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Content moderation system")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser(
        "server", help="Launch SGLang servers for LLM and/or embedding models"
    )
    server_parser.add_argument(
        "--llm", action="store_true", help="Launch the LLM server"
    )
    server_parser.add_argument(
        "--embedding", action="store_true", help="Launch the embedding server"
    )
    server_parser.add_argument(
        "--llm-port", type=int, default=8899, help="Port for the LLM server"
    )
    server_parser.add_argument(
        "--emb-port", type=int, default=8890, help="Port for the embedding server"
    )
    server_parser.add_argument(
        "--llm-model",
        type=str,
        default="microsoft/Phi-3.5-mini-instruct",
        help="Model name for the LLM server",
    )
    server_parser.add_argument(
        "--emb-model",
        type=str,
        default="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        help="Model name for the embedding server",
    )

    # Setup DB command
    setup_db_parser = subparsers.add_parser(
        "setup-db", help="Set up the vector database for content moderation"
    )
    setup_db_parser.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Path to the training data file",
    )
    setup_db_parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the vector database",
    )
    setup_db_parser.add_argument(
        "--embedding-url",
        type=str,
        default="http://localhost:8890/v1",
        help="URL of the embedding server",
    )

    # Moderate command
    moderate_parser = subparsers.add_parser(
        "moderate", help="Classify text content into moderation categories"
    )
    moderate_parser.add_argument(
        "text", nargs="?", help="Text to classify (if not provided, reads from stdin)"
    )
    moderate_parser.add_argument(
        "--vector-db",
        type=str,
        default=None,
        help="Path to the vector database",
    )
    moderate_parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG enhancement",
    )

    return parser.parse_args()


async def main():
    """Main function."""
    args = parse_args()

    if args.command == "server":
        # Convert args to sys.argv for server_main
        sys.argv = [sys.argv[0]]
        if args.llm:
            sys.argv.append("--llm")
        if args.embedding:
            sys.argv.append("--embedding")
        if args.llm_port:
            sys.argv.extend(["--llm-port", str(args.llm_port)])
        if args.emb_port:
            sys.argv.extend(["--emb-port", str(args.emb_port)])
        if args.llm_model:
            sys.argv.extend(["--llm-model", args.llm_model])
        if args.emb_model:
            sys.argv.extend(["--emb-model", args.emb_model])

        # Run the server
        server_main()

    elif args.command == "setup-db":
        # Run the setup-db command
        await setup_db_main()

    elif args.command == "moderate":
        # Run the moderate command
        await moderator_main()

    else:
        # No command specified, print help
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
