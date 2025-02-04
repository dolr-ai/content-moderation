import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
import json
from datetime import datetime
import time
import gc
import os


class KoalaAIClassifier:
    # Model-specific label mappings
    ORIGINAL_LABELS = {
        "S": "sexual",
        "H": "hate",
        "V": "violence",
        "HR": "harassment",
        "SH": "self-harm",
        "S3": "sexual/minors",
        "H2": "hate/threatening",
        "V2": "violence/graphic",
        "OK": "clean",
    }

    # Standardized category mapping
    LABEL_TO_PRIMARY = {
        "S": "nsfw_content",
        "H": "hate_or_discrimination",
        "V": "violence_or_threats",
        "HR": "offensive_language",
        "SH": "violence_or_threats",
        "S3": "nsfw_content",
        "H2": "hate_or_discrimination",
        "V2": "violence_or_threats",
        "OK": "clean",
    }

    PRIMARY_CATEGORY_MAP = {
        "clean": 0,
        "hate_or_discrimination": 1,
        "violence_or_threats": 2,
        "offensive_language": 3,
        "nsfw_content": 4,
        "spam_or_scams": 5,
    }

    def __init__(self, model_name: str = "KoalaAI/Text-Moderation", device: str = None):
        """Initialize the text moderation classifier."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def predict_single(self, text: str) -> Dict[str, Union[str, Dict]]:
        """
        Run inference on a single text input.
        """
        try:
            inputs = self.tokenizer(
                text, truncation=True, max_length=512, padding=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

            pred_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0][pred_idx].item()

            # Get original model label
            model_label = self.id2label[pred_idx]

            # Map to primary category
            primary_category = self.LABEL_TO_PRIMARY[model_label]

            # Get probabilities for each label
            label_probs = {}
            for idx, label in self.id2label.items():
                label_probs[label] = probabilities[0][idx].item()

            return {
                "text": text,
                "model_prediction": {
                    "label": model_label,
                    "meaning": self.ORIGINAL_LABELS[model_label],
                    "confidence": confidence,
                    "all_probabilities": label_probs,
                },
                "primary_prediction": {
                    "category": primary_category,
                    "category_id": self.PRIMARY_CATEGORY_MAP[primary_category],
                    "confidence": confidence,
                },
            }

        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return {
                "text": text,
                "model_prediction": {
                    "label": "error",
                    "meaning": "error",
                    "confidence": 0.0,
                    "all_probabilities": {},
                },
                "primary_prediction": {
                    "category": "error",
                    "category_id": -1,
                    "confidence": 0.0,
                },
            }

    def load_benchmark_data(self, file_path: str, sample: int = None) -> List[Dict]:
        """
        Load benchmark data from a JSONL file using pandas.

        Args:
            file_path: Path to the JSONL file containing benchmark data

        Returns:
            List of dictionaries containing benchmark samples
        """
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

    def run_benchmark(
        self,
        benchmark_data: List[Dict],
        batch_size: int = 32,
        output_dir: str = "benchmark_results",
        model_name: str = "koala",
    ):
        """
        Run benchmark on the loaded data and save results.
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir, f"{model_name}_benchmark_results_{timestamp}.jsonl"
        )

        print(f"Starting benchmark with {len(benchmark_data)} samples")
        print(f"Results will be saved to: {output_file}")

        total_processing_time = 0
        results = []

        for batch_num in range(0, len(benchmark_data), batch_size):
            batch_start_time = time.time()

            # Process batch
            batch = benchmark_data[batch_num : batch_num + batch_size]
            batch_texts = [sample["text"] for sample in batch]

            # Run predictions
            predictions = []
            for text in tqdm(
                batch_texts,
                desc=f"Processing batch {batch_num//batch_size + 1}/{len(benchmark_data)//batch_size + 1}",
            ):
                pred_start_time = time.time()
                prediction = self.predict_single(text)
                processing_time = time.time() - pred_start_time

                predictions.append(
                    {"prediction": prediction, "processing_time": processing_time}
                )

            batch_processing_time = time.time() - batch_start_time
            total_processing_time += batch_processing_time

            # Save batch results
            for sample, pred_info in zip(batch, predictions):
                result = {
                    "text_id": sample.get("text_id", -1),
                    "text": sample["text"],
                    "actual_category": sample.get("moderation_category", "unknown"),
                    "raw_llm_response": {
                        "label": pred_info["prediction"]["model_prediction"]["label"],
                        "meaning": pred_info["prediction"]["model_prediction"][
                            "meaning"
                        ],
                        "confidence": pred_info["prediction"]["model_prediction"][
                            "confidence"
                        ],
                        "all_probabilities": pred_info["prediction"][
                            "model_prediction"
                        ]["all_probabilities"],
                    },
                    "processed_response": {
                        "scores": {
                            cat: 0.0 for cat in self.PRIMARY_CATEGORY_MAP.keys()
                        },
                        "predicted_category": pred_info["prediction"][
                            "primary_prediction"
                        ]["category"],
                        "predicted_score": pred_info["prediction"][
                            "primary_prediction"
                        ]["confidence"],
                    },
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": pred_info["processing_time"],
                    "batch_num": batch_num // batch_size,
                    "batch_processing_time": batch_processing_time,
                }

                # Update scores based on model predictions
                for label, prob in pred_info["prediction"]["model_prediction"][
                    "all_probabilities"
                ].items():
                    primary_cat = self.LABEL_TO_PRIMARY[label]
                    result["processed_response"]["scores"][primary_cat] = max(
                        result["processed_response"]["scores"][primary_cat], prob
                    )

                # Write result to file
                with open(output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                results.append(result)

            # Clear CUDA cache if using GPU
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        print(f"\nBenchmark completed!")
        print(f"Total processing time: {total_processing_time:.2f} seconds")
        print(
            f"Average processing time per sample: {total_processing_time/len(benchmark_data):.2f} seconds"
        )

        # Show some sample results
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

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Run batch inference on multiple texts.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                results.append(self.predict_single(text))
        return results


if __name__ == "__main__":
    # Example usage
    model_name = "KoalaAI/Text-Moderation"
    device = "cuda"
    classifier = KoalaAIClassifier(model_name=model_name, device=device)
    benchmark_data = classifier.load_benchmark_data("./benchmark_v1.jsonl", sample=100)
    results = classifier.run_benchmark(benchmark_data, batch_size=256)
