# https://arxiv.org/pdf/2312.06674
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-3/
# https://huggingface.co/meta-llama/Llama-Guard-3-1B
# https://huggingface.co/meta-llama/Meta-Llama-Guard-3-8B
# https://the-ai-alliance.github.io/trust-safety-user-guide/exploring/mlcommons-taxonomy-hazards/

import torch
import random
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


class LlamaGuardModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-Guard-3-1B",
        device: str = None,
        use_flash_attention: bool = True,
        quantization: str = None,  # Options: '4bit', '8bit', None
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """Initialize the Llama Guard model and tokenizer.

        Args:
            model_name: Name or path of the model (1B or 8B variant)
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

    def test_inference(self, prompt="Hello, how are you?", max_new_tokens=100) -> None:
        """Test if the model is loaded correctly and can generate responses."""
        try:
            test_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTask: Check if there is unsafe content in 'User' messages in conversations according our safety policy.\n\n<BEGIN CONVERSATION>\n\nUser: {prompt}\n\n<END CONVERSATION>\n\nProvide your safety assessment for ONLY THE LAST User message in the above conversation:\n - First line must read 'safe' or 'unsafe'.\n - If unsafe, a second line must include a comma-separated list of violated categories. <|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            response = self.generate_response(
                test_prompt, max_new_tokens=max_new_tokens
            )
            print("\nModel test inference:")
            print(f"Input: '{prompt}'")
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


class LlamaGuardModerator:
    CATEGORIES = {
        "S1": "Violent Crimes.",
        "S2": "Non-Violent Crimes.",
        "S3": "Sex Crimes.",
        "S4": "Child Exploitation.",
        "S5": "Defamation.",
        "S6": "Specialized Advice.",
        "S7": "Privacy.",
        "S8": "Intellectual Property.",
        "S9": "Indiscriminate Weapons.",
        "S10": "Hate.",
        "S11": "Self-Harm.",
        "S12": "Sexual Content.",
        "S13": "Elections.",
    }

    LLAMAGUARD_TO_PLATFORM_CATEGORY_MAPPING = {
        # Maps Llama Guard's violent crimes (S1), hate (S10), and indiscriminate weapons (S9) suicide & self-harm (S11)
        # to platform's violence_or_threats category
        "S1": "violence_or_threats",  # Violent crimes (terrorism, murder, assault, etc.)
        "S9": "violence_or_threats",  # Indiscriminate weapons (chemical, biological, nuclear weapons)
        "S11": "violence_or_threats",  # Suicide & self-harm content
        # Maps Llama Guard's hate (S10) to platform's hate_or_discrimination category
        "S10": "hate_or_discrimination",  # Hate speech based on protected characteristics
        # Maps Llama Guard's sex-related crimes (S3), child exploitation (S4),
        # and sexual content (S12) to platform's nsfw_content category
        "S3": "nsfw_content",  # Sex-related crimes
        "S4": "nsfw_content",  # Child sexual exploitation - requires immediate escalation
        "S12": "nsfw_content",  # Sexual content/erotica
        # Maps Llama Guard's non-violent crimes (S2) and specialized advice (S6)
        # to platform's spam_or_scams category
        "S2": "spam_or_scams",  # Non-violent crimes (fraud, scams, theft)
        # Maps Llama Guard's defamation (S5), privacy (S7),
        # to platform's offensive_language category
        "S5": "offensive_language",  # Defamation - verifiably false statements harming reputation
        # The following Llama Guard categories require special handling as they don't map
        # directly to platform categories and may need additional logic:
        "S6": "spam_or_scams",  # Specialized advice - mapped to spam_or_scams but may need review
        "S7": "offensive_language",  # Privacy violations - mapped to offensive_language but may need review
        "S8": "spam_or_scams",  # Intellectual property - mapped to spam_or_scams but may need review
        "S13": "spam_or_scams",  # Election misinformation - mapped to spam_or_scams but may need review
    }

    def __init__(self, model: LlamaGuardModel = None):
        """Initialize the content moderator with a LlamaGuardModel instance."""
        self.model = model or LlamaGuardModel()

    def _create_prompt(self, text: str) -> str:
        """Create the moderation prompt following Llama Guard format."""
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{chr(10).join(f"{key}: {desc}" for key, desc in self.CATEGORIES.items())}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {text}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories. <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def _parse_response(self, response: str) -> Dict[str, Union[str, float]]:
        """Parse the generated response to extract safety assessment and categories."""
        # Extract the response part after the last assistant header
        response_text = response.split("<|start_header_id|>assistant<|end_header_id|>")[
            -1
        ].strip()

        # Clean up response by removing all <|eot_id|> tags and extra whitespace
        response_text = re.sub(r"<\|eot_id\|>", "", response_text).strip()

        # Split into lines and clean up
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]

        # Extract the main safety assessment from first line
        is_safe = lines[0].lower() == "safe" if lines else True

        # Extract violated categories if unsafe
        violated_categories = []
        if not is_safe and len(lines) > 1:
            # Split categories on comma and clean up each category
            categories = [cat.strip() for cat in lines[1].split(",")]
            # Only keep valid category codes
            violated_categories = [cat for cat in categories if cat in self.CATEGORIES]
            violated_llamaguard_categories = [
                self.CATEGORIES.get(cat) for cat in categories if cat in self.CATEGORIES
            ]

            # Map Llama Guard categories to standard categories
            violated_categories = self._map_categories(violated_categories)

        return {
            "is_safe": is_safe,
            "violated_categories": violated_categories,
            "violated_llamaguard_categories": violated_llamaguard_categories,
            "confidence": 0.9 if is_safe else 0.8,
            "raw_response": response,
        }

    def _map_categories(self, llama_guard_categories: List[str]) -> List[str]:
        """Map Llama Guard categories to standard categories."""
        category_mapping = self.LLAMAGUARD_TO_PLATFORM_CATEGORY_MAPPING

        mapped = set()
        for category in llama_guard_categories:
            if category in category_mapping:
                mapped.add(category_mapping[category])

        return list(mapped)

    def predict_single(self, text: str) -> Dict[str, Union[str, Dict]]:
        """Run inference on a single text input."""
        try:
            prompt = self._create_prompt(text)
            response = self.model.generate_response(prompt)
            parsed = self._parse_response(response)

            return {
                "text": text,
                "raw_response": response,
                "is_safe": parsed["is_safe"],
                "violated_categories": parsed["violated_categories"],
                "confidence": parsed["confidence"],
            }

        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                "text": text,
                "raw_response": str(e),
                "is_safe": False,
                "violated_categories": ["error"],
                "confidence": 0.0,
            }

    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_new_tokens=200,
        temperature=0.1,
        do_sample=True,
        skip_special_tokens=False,
    ) -> List[Dict]:
        """Run batch inference on multiple texts."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            prompts = [self._create_prompt(text) for text in batch_texts]

            inputs = self.model.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.model.tokenizer.pad_token_id,
                )

            responses = self.model.tokenizer.batch_decode(
                outputs, skip_special_tokens=skip_special_tokens
            )

            batch_results = [self._parse_response(response) for response in responses]

            for text, result in zip(batch_texts, batch_results):
                results.append(
                    {
                        "text": text,
                        "raw_response": result["raw_response"],
                        "is_safe": result["is_safe"],
                        "violated_categories": result["violated_categories"],
                        "confidence": result["confidence"],
                    }
                )

            if self.model.device == "cuda":
                torch.cuda.empty_cache()

        return results

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

    def run_benchmark(
        self,
        benchmark_data: List[Dict],
        batch_size: int = 32,
        output_dir: str = "benchmark_results",
        model_name: str = "llamaguard",
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

        for batch_num in tqdm(
            range(0, len(benchmark_data), batch_size), desc="Processing batches"
        ):
            batch_start_time = time.time()

            batch = benchmark_data[batch_num : batch_num + batch_size]
            batch_texts = [sample["text"] for sample in batch]

            pred_start_time = time.time()
            predictions = self.predict_batch(
                batch_texts,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                skip_special_tokens=skip_special_tokens,
            )
            batch_processing_time = time.time() - pred_start_time

            total_processing_time += batch_processing_time

            for sample, prediction in zip(batch, predictions):
                result = {
                    "text_id": sample.get("text_id", -1),
                    "text": sample["text"],
                    "actual_category": sample.get("moderation_category", "unknown"),
                    "raw_llm_response": prediction["raw_response"],
                    "processed_response": {
                        "is_safe": prediction["is_safe"],
                        "violated_categories": prediction["violated_categories"],
                        "confidence": prediction["confidence"],
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
            print(f"Is safe: {result['processed_response']['is_safe']}")
            print(
                f"Violated categories: {result['processed_response']['violated_categories']}"
            )
            print(f"Confidence: {result['processed_response']['confidence']:.3f}")

        return results

    def show_example_prompt(self, text: str = "This is a test message") -> None:
        """Display an example prompt with all components."""
        print("=== EXAMPLE CONTENT MODERATION PROMPT ===\n")
        prompt = self._create_prompt(text)
        print(prompt)
        print("\nExpected Response Format:")
        print("-" * 50)
        print("safe")
        print("or")
        print("unsafe")
        print("S1, S10")


# Example usage
if __name__ == "__main__":
    llama_model_params = "1B"  # or "8B"
    batch_size = 16

    if llama_model_params == "1B":
        print("Loading Llama-Guard-3-1B model...")
        llama_model = LlamaGuardModel(
            model_name="meta-llama/Llama-Guard-3-1B",
            use_flash_attention=True,
            quantization="8bit",
            torch_dtype=torch.bfloat16,
        )
    else:
        print("Loading Llama-Guard-3-8B model...")
        llama_model = LlamaGuardModel(
            model_name="meta-llama/Meta-Llama-Guard-3-8B",
            use_flash_attention=True,
            quantization="8bit",
            torch_dtype=torch.bfloat16,
        )

    moderator = LlamaGuardModerator(llama_model)

    # Load benchmark data
    benchmark_data = moderator.load_benchmark_data(
        "./benchmark_v1.jsonl",  # Update path to your benchmark file
        sample=100,  # Optional: reduce sample size for testing
    )

    # Run benchmark
    results = moderator.run_benchmark(
        benchmark_data=benchmark_data,
        batch_size=batch_size,
        output_dir=f"benchmark_results-{llama_model_params}",
        model_name=f"llamaguard-{llama_model_params}",
        debug=False,
        write_batch_size=10,
    )

    # Check random results
    def check_random_results(output_dir: str, model_name: str, num_samples: int = 5):
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

        results = []
        with open(file_path, "r") as f:
            for line in f:
                results.append(json.loads(line))

        sample_size = min(num_samples, len(results))
        random_samples = random.sample(results, sample_size)

        for i, result in enumerate(random_samples, 1):
            print(f"\nSample {i}:")
            print(f"Text: {result['text'][:200]}...")
            print(f"Actual category: {result['actual_category']}")
            print(f"Is safe: {result['processed_response']['is_safe']}")
            print(
                f"Violated categories: {result['processed_response']['violated_categories']}"
            )
            print(f"Confidence: {result['processed_response']['confidence']:.2f}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            print("-" * 80)

    check_random_results(
        output_dir=f"benchmark_results-{llama_model_params}",
        model_name=f"llamaguard-{llama_model_params}",
        num_samples=5,
    )
