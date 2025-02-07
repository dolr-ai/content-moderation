# https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from typing import List, Dict, Union
from tqdm import tqdm
import json
from datetime import datetime
import time
import gc
import os
import pandas as pd
import re


class Llama32Model:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        device: str = None,
        use_flash_attention: bool = True,
        quantization: str = None,  # Options: '4bit', '8bit', None
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the Llama 3.2 model and tokenizer.

        Args:
            model_name: Name or path of the model (1B or 3B variant)
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_flash_attention: Whether to use flash attention for faster inference
            quantization: Quantization type ('4bit', '8bit', or None)
            torch_dtype: Torch data type for model weights
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Configure model loading kwargs
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        # Add flash attention if requested and available
        if use_flash_attention and self.device == "cuda":
            try:
                if is_flash_attn_2_available():
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Flash Attention 2.0 enabled")
                else:
                    print(
                        "Flash Attention not available, falling back to standard attention"
                    )
            except ImportError:
                print(
                    "Flash Attention import failed, falling back to standard attention"
                )

        # Configure quantization
        if quantization:
            try:
                if quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    print("Using 4-bit quantization")
                elif quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch_dtype,
                        bnb_8bit_use_double_quant=True,
                    )
                    print("Using 8-bit quantization")
            except ImportError:
                print("bitsandbytes not available, skipping quantization")
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )

        # Set padding token to EOS token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad_token to eos_token")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    def generate_response(
        self,
        prompt: str,
        skip_special_tokens=False,
        max_new_tokens: int = 200,
        temperature: float = 0.1,
        do_sample: bool = True,
    ) -> str:
        """Generate response from the model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )

        response = self.tokenizer.decode(
            outputs[0], skip_special_tokens=skip_special_tokens
        )
        return response

    def test_inference(
        self, prompt, max_new_tokens=100, skip_special_tokens=False
    ) -> None:
        """Test if the model is loaded correctly and can generate responses."""
        try:
            test_prompt = f"<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.\n<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
            response = self.generate_response(
                test_prompt,
                max_new_tokens=max_new_tokens,
                skip_special_tokens=skip_special_tokens,
            )

            print("\nModel test inference:")
            print("Input: 'Say Hello, I am working!'")
            print(f"Output: {response}")
            print("Model loaded and working successfully!\n")

        except Exception as e:
            print(f"\nError during test inference: {str(e)}")
            raise Exception("Model initialization failed!")

    def clear_memory(self):
        """Clear CUDA memory if using GPU."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()


class Llama32ContentModerator:
    CATEGORIES = {
        "hate_or_discrimination": "Content that expresses, incites, or promotes hate based on protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status, etc.), including hostile language, prejudiced views, or discriminatory ideology.",
        "violence_or_threats": "Content that threatens, depicts, or promotes violence, death, or physical harm. This includes graphic violence, threats of serious harm, instructions for violent acts, or glorification of violent behavior.",
        "offensive_language": "Harassing or severely inappropriate content that attacks, demeans, or abuses others, including hostile expressions, severe profanity, and malicious insults - even when not targeting protected characteristics.",
        "nsfw_content": "Material intended to arouse sexual excitement, explicit sexual content, or promotion of sexual services (excluding educational/wellness content). Special attention to any content involving minors, which requires immediate escalation.",
        "spam_or_scams": "Deceptive, fraudulent, or unsolicited content designed to mislead or exploit users, including phishing attempts, scam schemes, coordinated inauthentic behavior, and bulk unsolicited messages.",
        "clean": "Content that is allowed on platform and does not fall in the categories mentioned above.",
    }

    def __init__(self, model: Llama32Model = None):
        """Initialize the content moderator with a Llama32Model instance."""
        self.model = model or Llama32Model()

    def _create_prompt(self, text: str) -> str:
        """Create the moderation prompt."""
        categories_text = "\n".join(
            f"{i+1}. {cat} : {desc}"
            for i, (cat, desc) in enumerate(self.CATEGORIES.items())
        )

        return f"""<|start_header_id|>system<|end_header_id|>
You are a content moderation expert. Your task is to analyze content and categorize it into one of the following categories:

{categories_text}

For each piece of content, you should:
1. If no violations are found, classify as "clean"
2. Provide your confidence level
3. Provide one or two line explanation

Please format your response exactly as:
Category: [exact category_name]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [short 1/2 line explanation]
<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Analyze this content:
{text}
<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""

    def _parse_response(self, response: str) -> Dict[str, Union[str, float]]:
        """Parse the generated response to extract category and confidence."""
        category_match = re.search(
            r"Category:\s*(\w+(?:_?\w+)*)", response, re.IGNORECASE
        )
        confidence_match = re.search(
            r"Confidence:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE
        )

        category = category_match.group(1).lower() if category_match else "clean"
        confidence = confidence_match.group(1).upper() if confidence_match else "LOW"

        if category not in self.CATEGORIES:
            print(
                f"Warning: Invalid category '{category}' detected, defaulting to 'clean'"
            )
            category = "clean"

        confidence_scores = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
        score = confidence_scores.get(confidence, 0.3)

        return {"category": category, "confidence": score, "raw_response": response}

    def predict_single(self, text: str) -> Dict[str, Union[str, Dict]]:
        """Run inference on a single text input."""
        try:
            prompt = self._create_prompt(text)
            response = self.model.generate_response(prompt)
            parsed = self._parse_response(response)

            return {
                "text": text,
                "raw_response": response,
                "category": parsed["category"],
                "confidence": parsed["confidence"],
            }

        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                "text": text,
                "raw_response": str(e),
                "category": "error",
                "confidence": 0.0,
            }

    def load_benchmark_data(self, file_path: str, sample=None) -> List[Dict]:
        """Load benchmark data from a JSONL file using pandas."""
        print(f"Loading benchmark data from {file_path}")

        df = pd.read_json(file_path, lines=True)

        if sample:
            df = df.head(sample)

        if "text_id" not in df.columns:
            df["text_id"] = range(len(df))

        data = df.to_dict("records")

        print(f"Loaded {len(data)} benchmark samples")
        print(f"Columns found: {', '.join(df.columns)}")

        if "moderation_category" in df.columns:
            print("\nCategory distribution:")
            print(df["moderation_category"].value_counts())
        return data

    def predict_batch(
        self,
        texts: List[str],
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        skip_special_tokens=False,
    ) -> List[Dict]:
        """Run batch inference on multiple texts."""
        # Create prompts for all texts
        prompts = [self._create_prompt(text) for text in texts]

        # Tokenize entire batch at once
        inputs = self.model.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        # Generate responses for entire batch
        with torch.no_grad():
            outputs = self.model.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.model.tokenizer.pad_token_id,
            )

        self.model.clear_memory()

        # Decode all responses in batch
        responses = self.model.tokenizer.batch_decode(
            outputs, skip_special_tokens=skip_special_tokens
        )

        # Parse all responses
        results = []
        for text, response in zip(texts, responses):
            parsed = self._parse_response(response)
            results.append(
                {
                    "text": text,
                    "raw_response": response,
                    "category": parsed["category"],
                    "confidence": parsed["confidence"],
                }
            )

        # Clear GPU memory
        if self.model.device == "cuda":
            torch.cuda.empty_cache()

        return results

    def run_benchmark(
        self,
        benchmark_data: List[Dict],
        batch_size: int = 32,
        output_dir: str = "benchmark_results",
        model_name: str = "llama32",
        debug: bool = False,
        write_batch_size: int = 50,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        skip_special_tokens=False,
    ):
        """Run benchmark on the loaded data and save results."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"{model_name}_benchmark_results_{timestamp}.jsonl"
        )

        print(f"Starting benchmark with {len(benchmark_data)} samples")
        print(f"Results will be saved to: {output_file}")

        total_processing_time = 0
        results = []
        results_to_write = []

        # Use tqdm to show progress for batch processing
        for batch_num in tqdm(
            range(0, len(benchmark_data), batch_size), desc="Processing batches"
        ):
            batch_start_time = time.time()

            batch = benchmark_data[batch_num : batch_num + batch_size]
            batch_texts = [sample["text"] for sample in batch]

            # Process entire batch at once
            pred_start_time = time.time()
            predictions = self.predict_batch(
                batch_texts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                skip_special_tokens=skip_special_tokens,
            )
            batch_processing_time = time.time() - pred_start_time

            total_processing_time += batch_processing_time

            for sample, prediction in zip(batch, predictions):
                category_scores = {cat: 0.0 for cat in self.CATEGORIES.keys()}
                category_scores[prediction["category"]] = prediction["confidence"]

                result = {
                    "text_id": sample.get("text_id", -1),
                    "text": sample["text"],
                    "actual_category": sample.get("moderation_category", "unknown"),
                    "raw_llm_response": prediction["raw_response"],
                    "processed_response": {
                        "scores": category_scores,
                        "predicted_category": prediction["category"],
                        "predicted_score": prediction["confidence"],
                    },
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": batch_processing_time / len(batch),
                    "batch_num": batch_num // batch_size,
                    "batch_processing_time": batch_processing_time,
                }

                results.append(result)
                results_to_write.append(result)

                if len(results_to_write) >= write_batch_size:
                    with open(output_file, "a") as f:
                        for r in results_to_write:
                            f.write(json.dumps(r) + "\n")
                    print(f"\nWrote {len(results_to_write)} results to file")
                    results_to_write = []

            if self.model.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        # Write any remaining results
        if results_to_write:
            with open(output_file, "a") as f:
                for r in results_to_write:
                    f.write(json.dumps(r) + "\n")
            print(f"\nWrote final {len(results_to_write)} results to file")

        print("\nBenchmark completed!")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(
            f"Average processing time per sample: {total_processing_time/len(benchmark_data):.2f} seconds"
        )

        print("\nSample results:")
        for i in range(min(3, len(results))):
            result = results[i]
            print(f"\nText: {result['text'][:100]}...")
            print(f"Actual category: {result['actual_category']}")
            print(
                f"Predicted category: {result['processed_response']['predicted_category']}"
            )
            print(f"Confidence: {result['processed_response']['predicted_score']:.3f}")

        return results

    def show_example_prompt(self, text: str = "This is a test message") -> None:
        """Display an example prompt with all components."""
        print("=== EXAMPLE CONTENT MODERATION PROMPT ===\n")

        # Create prompt with all policies
        prompt = self._create_prompt(text)

        print("System Message:")
        print("-" * 50)
        print(prompt)

        print("\nExpected Response Format:")
        print("-" * 50)
        print(
            """Category: clean
Confidence: HIGH"""
        )


# %%

llama_model_params = "1B"
batch_size = 16

if llama_model_params == "1B":
    # Load model and create moderator
    print("Loading Llama-3.2-1B-Instruct model...")

    # For 1B model
    llama_model = Llama32Model(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        use_flash_attention=True,
        quantization="8bit",
        torch_dtype=torch.bfloat16,
    )

elif llama_model_params == "3B":
    print("Loading Llama-3.2-3B-Instruct model...")
    # For 3B model
    llama_model = Llama32Model(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        use_flash_attention=True,
        quantization="8bit",
        torch_dtype=torch.bfloat16,
    )

moderator = Llama32ContentModerator(llama_model)

# Load benchmark data
benchmark_data = moderator.load_benchmark_data(
    "./benchmark_v1.jsonl",  # Update this path to your benchmark file
    sample=100,  # Optional: reduce sample size for testing
)

# Run benchmark
results = moderator.run_benchmark(
    benchmark_data=benchmark_data,
    batch_size=batch_size,
    output_dir=f"benchmark_results-{llama_model_params}",
    model_name=f"llama32-{llama_model_params}",
    debug=False,  # Set to False to disable debugging output
    write_batch_size=10,  # Set to 1 to write results after each sample
)

# %%
# randomly check samples from latest benchmark results
import random


def check_random_results(output_dir: str, model_name: str, num_samples: int = 5):
    # Get the latest results file
    result_files = [
        f
        for f in os.listdir(output_dir)
        if f.startswith(f"{model_name}_benchmark_results_")
    ]
    if not result_files:
        print(f"No result files found in {output_dir}")
        return

    latest_file = max(
        result_files, key=lambda x: os.path.getctime(os.path.join(output_dir, x))
    )
    file_path = os.path.join(output_dir, latest_file)

    print(f"\nChecking random samples from: {latest_file}")
    print("-" * 80)

    # Load results
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    # Randomly sample results
    sample_size = min(num_samples, len(results))
    random_samples = random.sample(results, sample_size)

    # Display samples
    for i, result in enumerate(random_samples, 1):
        print(f"\nSample {i}:")
        print(f"Text: {result['text'][:200]}...")
        print(f"Actual category: {result['actual_category']}")
        print(
            f"Predicted category: {result['processed_response']['predicted_category']}"
        )
        print(f"Confidence: {result['processed_response']['predicted_score']:.2f}")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print("-" * 80)


check_random_results(
    output_dir=f"benchmark_results-{llama_model_params}",
    model_name=f"llama32-{llama_model_params}",
    num_samples=5,
)
