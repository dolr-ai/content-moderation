from src.servers.moderation_server import run_server
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the moderation server")
    parser.add_argument("--vector-db-path", required=True, help="Path to vector database")
    parser.add_argument("--prompt-path", required=True, help="Path to prompt file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--embedding-url",
        default="http://localhost:8890/v1",
        help="URL for embedding API",
    )
    parser.add_argument(
        "--llm-url", default="http://localhost:8899/v1", help="URL for LLM API"
    )
    parser.add_argument(
        "--max-input-length", type=int, default=2000, help="Maximum input length"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    run_server(
        vector_db_path=args.vector_db_path,
        prompt_path=args.prompt_path,
        host=args.host,
        port=args.port,
        embedding_url=args.embedding_url,
        llm_url=args.llm_url,
        input_length=args.max_input_length,
        workers=args.workers,
        reload=args.reload,
    )