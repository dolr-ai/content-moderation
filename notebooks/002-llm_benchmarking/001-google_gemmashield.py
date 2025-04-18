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


class ShieldGemmaClassifier:
    PRIMARY_CATEGORY_MAP = {
        "clean": 0,
        "hate_or_discrimination": 1,
        "violence_or_threats": 2,
        "offensive_language": 3,
        "nsfw_content": 4,
        "spam_or_scams": 5,
    }

    SAFETY_POLICIES = {
        "hate_speech": {
            "policy": """* "No Hate or Discrimination": The prompt shall not contain or seek generation of content that:
              - Expresses, incites, or promotes hate based on protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status, etc.)
              - Promotes hostile language, prejudiced views, or discriminatory ideology
              - Dehumanizes or demeans individuals or groups based on identity
              - Promotes harmful stereotypes or biases
              - Denies or minimizes historical atrocities or systematic discrimination""",
            "primary_category": "hate_or_discrimination",
        },
        "violence": {
            "policy": """* "No Violence or Threats": The prompt shall not contain or seek generation of content that:
              - Threatens, depicts, or promotes violence, death, or physical harm
              - Provides instructions for violent acts or weapons (especially CBRNE)
              - Glorifies or endorses violent behavior or crimes
              - Promotes terrorism, extremist violence, or mass casualties
              - Encourages domestic abuse or other forms of targeted violence
              - Contains graphic descriptions of violence or gore""",
            "primary_category": "violence_or_threats",
        },
        "harassment": {
            "policy": """* "No Harassment or Offensive Content": The prompt shall not contain or seek generation of content that:
              - Harasses, intimidates, or bullies individuals or groups
              - Uses severe profanity or malicious insults
              - Mocks or demeans victims of violence or tragedy
              - Contains hostile or abusive language
              - Encourages coordinated harassment or abuse
              - Promotes harmful behaviors or dangerous practices""",
            "primary_category": "offensive_language",
        },
        "sexual": {
            "policy": """* "No NSFW or Exploitative Content": The prompt shall not contain or seek generation of content that:
              - Contains explicit sexual content or erotic material
              - Promotes sexual services or exploitation
              - Involves minors in any sexual context (requires immediate escalation)
              - Describes graphic sexual acts or extreme fetishes
              - Promotes non-consensual sexual content
              - Shares intimate content without consent""",
            "primary_category": "nsfw_content",
        },
        "harm": {
            "policy": """* "No Self-Harm or Dangerous Content": The prompt shall not contain or seek generation of content that:
              - Promotes or encourages suicide or self-harm
              - Provides instructions for dangerous behaviors
              - Promotes eating disorders or body dysmorphia
              - Encourages risky or life-threatening activities
              - Minimizes mental health issues
              - Shares detailed methods of self-harm""",
            "primary_category": "violence_or_threats",
        },
        "scams": {
            "policy": """* "No Scams or Deceptive Content": The prompt shall not contain or seek generation of content that:
              - Promotes fraudulent schemes or scams
              - Attempts phishing or identity theft
              - Engages in coordinated inauthentic behavior
              - Spreads malicious misinformation
              - Promotes financial exploitation
              - Contains unsolicited bulk messaging""",
            "primary_category": "spam_or_scams",
        },
    }

    def __init__(
        self,
        model_name: str = "google/shieldgemma-2b",
        device: str = None,
        use_flash_attention: bool = True,
    ):
        """Initialize the Shield Gemma classifier."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

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

        # Configure 8-bit quantization
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
            bnb_8bit_use_double_quant=True,
        )
        print("Using 8-bit quantization")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.vocab = self.tokenizer.get_vocab()
        self.yes_token_id = self.vocab["Yes"]
        self.no_token_id = self.vocab["No"]

    def _create_prompt(self, text: str, policy: str) -> str:
        return f"""You are a policy expert trying to help determine whether a user
prompt is in violation of the defined safety policies.

<start_of_turn>
Human Input: {text.strip()}
<end_of_turn>

Our safety principle is defined in the below:

{policy.strip()}

Does the human input violate the above principle? Your answer must start
with 'Yes' or 'No'. And then walk through step by step to be sure we answer
correctly."""

    def _get_violation_probability(self, text: str, policy: str) -> float:
        prompt = self._create_prompt(text, policy)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        selected_logits = logits[0, -1, [self.yes_token_id, self.no_token_id]]
        probabilities = torch.nn.functional.softmax(selected_logits, dim=0)

        return probabilities[0].item()

    def _determine_primary_category(
        self, policy_scores: Dict[str, float]
    ) -> Dict[str, Union[str, int, float]]:
        max_policy = max(policy_scores.items(), key=lambda x: x[1])
        policy_name, max_score = max_policy

        if max_score < 0.5:
            return {
                "category": "clean",
                "category_id": self.PRIMARY_CATEGORY_MAP["clean"],
                "confidence": 1 - max_score,
            }

        primary_category = self.SAFETY_POLICIES[policy_name]["primary_category"]
        return {
            "category": primary_category,
            "category_id": self.PRIMARY_CATEGORY_MAP[primary_category],
            "confidence": max_score,
        }

    def predict_single(self, text: str) -> Dict[str, Union[str, Dict]]:
        """Run inference on a single text input."""
        try:
            policy_scores = {}
            for policy_name, policy_info in self.SAFETY_POLICIES.items():
                score = self._get_violation_probability(text, policy_info["policy"])
                policy_scores[policy_name] = score

            primary_prediction = self._determine_primary_category(policy_scores)

            return {
                "text": text,
                "model_prediction": {
                    "policy_scores": policy_scores,
                    "confidence": primary_prediction["confidence"],
                },
                "primary_prediction": primary_prediction,
            }

        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                "text": text,
                "model_prediction": {"policy_scores": {}, "confidence": 0.0},
                "primary_prediction": {
                    "category": "error",
                    "category_id": -1,
                    "confidence": 0.0,
                },
            }

    def load_benchmark_data(self, file_path: str, sample=None) -> List[Dict]:
        """Load benchmark data from a JSONL file using pandas."""
        print(f"Loading benchmark data from {file_path}")

        # Read JSONL file using pandas
        df = pd.read_json(file_path, lines=True)

        if sample:
            df = df.head(sample)

        # Add text_id if not present
        if "text_id" not in df.columns:
            df["text_id"] = range(len(df))

        # Convert DataFrame to list of dictionaries
        data = df.to_dict("records")

        print(f"Loaded {len(data)} benchmark samples")
        print(f"Columns found: {', '.join(df.columns)}")

        # Print sample distribution if moderation_category exists
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
        prompts = []
        for text in texts:
            policy_prompts = []
            for policy_name, policy_info in self.SAFETY_POLICIES.items():
                prompt = self._create_prompt(text, policy_info["policy"])
                policy_prompts.append(prompt)
            prompts.extend(policy_prompts)

        # Tokenize entire batch at once
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate responses for entire batch
        with torch.no_grad():
            logits = self.model(**inputs).logits
            selected_logits = logits[:, -1, [self.yes_token_id, self.no_token_id]]
            probabilities = torch.nn.functional.softmax(selected_logits, dim=1)
            yes_probabilities = probabilities[:, 0].tolist()

        torch.cuda.empty_cache()
        gc.collect()

        # Process results for each text
        results = []
        for i in range(0, len(yes_probabilities), len(self.SAFETY_POLICIES)):
            text_probabilities = yes_probabilities[i : i + len(self.SAFETY_POLICIES)]
            policy_scores = {
                policy_name: score
                for (policy_name, _), score in zip(
                    self.SAFETY_POLICIES.items(), text_probabilities
                )
            }

            primary_prediction = self._determine_primary_category(policy_scores)

            results.append(
                {
                    "text": texts[i // len(self.SAFETY_POLICIES)],
                    "model_prediction": {
                        "policy_scores": policy_scores,
                        "confidence": primary_prediction["confidence"],
                    },
                    "primary_prediction": primary_prediction,
                }
            )

        # Clear GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        return results

    def run_benchmark(
        self,
        benchmark_data: List[Dict],
        batch_size: int = 32,
        output_dir: str = "benchmark_results",
        model_name: str = "shieldgemma",
        debug: bool = False,
        write_batch_size: int = 50,
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
            batch = benchmark_data[batch_num : batch_num + batch_size]
            batch_texts = [sample["text"] for sample in batch]

            # Process entire batch at once
            pred_start_time = time.time()
            predictions = self.predict_batch(batch_texts)
            batch_processing_time = time.time() - pred_start_time

            total_processing_time += batch_processing_time

            for sample, prediction in zip(batch, predictions):
                result = {
                    "text_id": sample.get("text_id", -1),
                    "text": sample["text"],
                    "actual_category": sample.get("moderation_category", "unknown"),
                    "raw_llm_response": {
                        "policy_scores": prediction["model_prediction"][
                            "policy_scores"
                        ],
                        "confidence": prediction["model_prediction"]["confidence"],
                    },
                    "processed_response": {
                        "scores": {
                            cat: 0.0 for cat in self.PRIMARY_CATEGORY_MAP.keys()
                        },
                        "predicted_category": prediction["primary_prediction"][
                            "category"
                        ],
                        "predicted_score": prediction["primary_prediction"][
                            "confidence"
                        ],
                    },
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": batch_processing_time / len(batch),
                    "batch_num": batch_num // batch_size,
                    "batch_processing_time": batch_processing_time,
                }

                # Update scores based on policy predictions
                for policy, score in prediction["model_prediction"][
                    "policy_scores"
                ].items():
                    primary_cat = self.SAFETY_POLICIES[policy]["primary_category"]
                    result["processed_response"]["scores"][primary_cat] = max(
                        result["processed_response"]["scores"][primary_cat], score
                    )

                results.append(result)
                results_to_write.append(result)

                if len(results_to_write) >= write_batch_size:
                    with open(output_file, "a") as f:
                        for r in results_to_write:
                            f.write(json.dumps(r) + "\n")
                    print(f"\nWrote {len(results_to_write)} results to file")
                    results_to_write = []

        # Write any remaining results
        if results_to_write:
            with open(output_file, "a") as f:
                for r in results_to_write:
                    f.write(json.dumps(r) + "\n")
            print(f"\nWrote final {len(results_to_write)} results to file")

        print(f"\nBenchmark completed!")
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


if __name__ == "__main__":
    # Example usage
    model_name = "google/shieldgemma-2b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = ShieldGemmaClassifier(
        model_name=model_name, device=device, use_flash_attention=True
    )
    benchmark_data = classifier.load_benchmark_data("./benchmark_v1.jsonl", sample=100)
    results = classifier.run_benchmark(benchmark_data, batch_size=4)
