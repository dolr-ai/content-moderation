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
from src.performance.performance_visualizer import PerformanceVisualizer
from src.cli.parsers import (
    add_server_parser,
    add_vectordb_parser,
    add_moderation_parser,
    add_moderation_server_parser,
    add_performance_parser,
    add_perf_visualization_parser,
)

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

    # Add parsers for different commands
    add_server_parser(subparsers)
    add_vectordb_parser(subparsers)
    add_moderation_parser(subparsers)
    add_moderation_server_parser(subparsers)
    add_performance_parser(subparsers)
    add_perf_visualization_parser(subparsers)

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
        start_embedding=start_embedding, start_llm=start_llm, emb_timeout=args.emb_timeout, llm_timeout=args.llm_timeout
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
            result = system.classify_text(
                args.text,
                num_examples=args.num_examples,
                max_input_length=args.max_input_length,
            )
            print(json.dumps(result, indent=2))
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
        input_length=args.max_input_length,
    )


def run_performance_command(args):
    """Run performance testing command"""
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
            concurrency_levels=args.concurrency_levels,
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

        # Run visualization if not skipped
        if not args.skip_visualization:
            logger.info("Generating visualizations...")

            # Determine which files to use
            results_file = None
            scaling_report_file = None

            if args.run_scaling_test:
                scaling_report_file = Path(args.output_dir) / "scaling_report.json"
            else:
                results_file = Path(args.output_dir) / "performance_results.json"

            # Create visualization output directory
            viz_output_dir = Path(args.output_dir) / "visualizations"

            # Run visualization
            viz_result = visualize_performance_results(
                results_file=results_file,
                scaling_report_file=scaling_report_file,
                output_dir=viz_output_dir,
                generate_report=True,
            )

            if "error" in viz_result:
                logger.error(f"Visualization generation failed: {viz_result['error']}")
            else:
                logger.info("Visualizations generated successfully")

                if "plots" in viz_result and viz_result["plots"]:
                    logger.info(
                        f"Generated plots: {', '.join(viz_result['plots'].keys())}"
                    )

                if "report" in viz_result and viz_result["report"]:
                    logger.info(
                        f"Visualization report saved to: {viz_result['report']}"
                    )

        # Print report path from performance test
        if "report_path" in results:
            logger.info(f"Performance report saved to: {results['report_path']}")

        return True
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        return False


def run_perf_visualization_command(args):
    """Run visualization command"""
    # Validate inputs
    if not args.results_file and not args.scaling_report_file:
        logger.error(
            "No results files specified. Use --results-file or --scaling-report-file."
        )
        return False

    try:
        # Create visualizer instance
        visualizer = PerformanceVisualizer(
            results_file=args.results_file,
            scaling_report_file=args.scaling_report_file,
            output_dir=args.output_dir,
        )

        # Generate plots
        plots = visualizer.generate_all_plots()

        # Generate report if requested
        report_path = ""
        if not args.no_report:
            report_path = visualizer.generate_markdown_report()

        # Report success
        logger.info("Visualization completed successfully")

        if plots:
            logger.info(f"Generated plots: {', '.join(plots.keys())}")

        if report_path:
            logger.info(f"Generated report: {report_path}")

        return True
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return False


def visualize_performance_results(
    results_file=None, scaling_report_file=None, output_dir=None, generate_report=True
):
    """
    Helper function to visualize performance results

    Args:
        results_file: Path to performance results JSON file
        scaling_report_file: Path to scaling report JSON file
        output_dir: Directory to save visualizations
        generate_report: Whether to generate a markdown report

    Returns:
        Dictionary with visualization results
    """
    try:
        # Create visualizer instance
        visualizer = PerformanceVisualizer(
            results_file=results_file,
            scaling_report_file=scaling_report_file,
            output_dir=output_dir,
        )

        # Generate plots
        plots = visualizer.generate_all_plots()

        # Generate report if requested
        report_path = ""
        if generate_report:
            report_path = visualizer.generate_markdown_report()

        return {"plots": plots, "report": report_path}
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        return {"error": str(e)}


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
    elif args.command == "visualize":
        success = run_perf_visualization_command(args)
    else:
        print(
            "No command specified. Use server, vectordb, moderate, moderation-server, performance, or visualize."
        )
        success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
