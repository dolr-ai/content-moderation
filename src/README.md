# Content Moderation System

A RAG-enhanced content moderation system for classifying text into moderation categories.

## Overview

This system uses a combination of LLM inference and RAG (Retrieval-Augmented Generation) to classify text content into moderation categories. It consists of several components:

1. **SGLang Servers**: Servers for LLM inference and embedding generation
2. **Vector Database**: FAISS-based vector database for similarity search
3. **Prompt Templates**: Jinja2-based templates for generating prompts
4. **Inference Engine**: Core moderation functionality

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/content-moderation.git
   cd content-moderation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the system by creating a `dev_config.yml` file:
   ```yaml
   local:
     PROJECT_ROOT: /path/to/content-moderation
     DATA_ROOT: /path/to/content-moderation/data

   tokens:
     HF_TOKEN: your_huggingface_token
     OPENAI_API_KEY: your_openai_api_key
   ```

## Usage

### Starting the Servers

To start the SGLang servers for LLM and embedding models:

```bash
python -m src.main server --llm --embedding
```

Options:
- `--llm`: Start the LLM server
- `--embedding`: Start the embedding server
- `--llm-port PORT`: Port for the LLM server (default: 8899)
- `--emb-port PORT`: Port for the embedding server (default: 8890)
- `--llm-model MODEL`: Model name for the LLM server (default: microsoft/Phi-3.5-mini-instruct)
- `--emb-model MODEL`: Model name for the embedding server (default: Alibaba-NLP/gte-Qwen2-1.5B-instruct)

### Setting Up the Vector Database

To set up the vector database from training data:

```bash
python -m src.main setup-db --training-data /path/to/training_data.csv
```

Options:
- `--training-data PATH`: Path to the training data file (required)
- `--output-path PATH`: Path to save the vector database (default: data/vector_db)
- `--embedding-url URL`: URL of the embedding server (default: http://localhost:8890/v1)

### Moderating Content

To classify text content:

```bash
python -m src.main moderate "This is some text to moderate"
```

Options:
- `--vector-db PATH`: Path to the vector database (default: data/vector_db)
- `--no-rag`: Disable RAG enhancement
- `--output PATH`: Output file for the result (JSON format)

You can also pipe text to the command:

```bash
echo "This is some text to moderate" | python -m src.main moderate
```

## Architecture

The system is organized into the following components:

- `src/config/`: Configuration management
- `src/prompts/`: Prompt templates using Jinja2
- `src/server/`: SGLang server management
- `src/vector_db/`: Vector database for similarity search
- `src/inference/`: Core moderation functionality

## Moderation Categories

The system classifies text into the following categories:

1. `hate_or_discrimination`: Content targeting protected characteristics with negative intent/prejudice
2. `violence_or_threats`: Content that threatens, depicts, or promotes violence
3. `offensive_language`: Hostile or inappropriate content WITHOUT targeting protected characteristics
4. `nsfw_content`: Explicit sexual content or material intended to arouse
5. `spam_or_scams`: Deceptive or unsolicited content designed to mislead
6. `clean`: Content that is allowed and doesn't fall into above categories

