#!/usr/bin/env python3
"""
Entry point for content moderation system

This module serves as the main entry point for the content moderation system.
It integrates the server management, vector database, and moderation components.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.main import main as cli_main

if __name__ == "__main__":
    sys.exit(cli_main())
