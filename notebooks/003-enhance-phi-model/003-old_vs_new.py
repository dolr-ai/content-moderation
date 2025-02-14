import os
import json
from datetime import datetime
import time
from tqdm import tqdm
import pandas as pd
import sglang as sgl
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

API_KEY = "None"

client = openai.Client(
    base_url="http://<add-your-internal-ip>:8890/v1",
    api_key=API_KEY,
)


SYSTEM_PROMPT = """
You are a content moderation expert responsible for accurately classifying content into predefined categories. Your analysis should be precise, context-aware, and avoid over-classification.

CATEGORIES AND GUIDELINES:

1. hate_or_discrimination
- REQUIRES: Explicit targeting of protected groups
- MUST HAVE: Slurs, dehumanizing language, or clear discriminatory intent
- EXAMPLES:
  VIOLATES: "All [racial slur]s should die"
  VIOLATES: "Women don't deserve rights"
  ALLOWED: "I disagree with immigration policies" (political opinion)
  ALLOWED: "This tax system is unfair" (policy criticism)

2. violence_or_threats
- REQUIRES: Credible expression of intent to harm
- MUST HAVE: Specific threats or detailed violent content
- EXAMPLES:
  VIOLATES: "I will find you and hurt you"
  VIOLATES: "Let's burn down their houses"
  ALLOWED: "I'm so angry I could scream" (expression of emotion)
  ALLOWED: "Fighting game tournament tonight" (gaming context)

3. offensive_language
- REQUIRES: Profanity or crude language
- MUST HAVE: Insulting or inappropriate content
- EXAMPLES:
  VIOLATES: "This is complete bullsh*t"
  VIOLATES: "You're such an idiot"
  ALLOWED: "This service is terrible" (negative but not offensive)
  ALLOWED: "I strongly disagree" (civil disagreement)

4. nsfw_content
- REQUIRES: Sexual content or nudity
- MUST HAVE: Explicit or suggestive material
- EXAMPLES:
  VIOLATES: "Let's have hot sex tonight"
  VIOLATES: Explicit descriptions of sexual acts
  ALLOWED: "Medical examination required" (clinical context)
  ALLOWED: "Beautiful beach photo" (non-sexual context)

5. spam_or_scams
- REQUIRES: Unsolicited commercial content OR deceptive intent
- MUST HAVE: One or more of: urgency, requests for action/information, too-good-to-be-true offers
- EXAMPLES:
  VIOLATES: "URGENT: Your account will be suspended unless..."
  VIOLATES: "Make $5000/day working from home!"
  ALLOWED: "Our store is having a sale" (legitimate marketing)
  ALLOWED: "Please respond to my email" (normal communication)

6. clean
- Default category when content doesn't meet criteria for other categories
- Can include negative or controversial content that doesn't violate specific rules

CLASSIFICATION INSTRUCTIONS:

1. Read the content carefully and consider full context
2. Check against MUST HAVE criteria for each category
3. If in doubt between categories, use these priority rules:
   - Hate speech > Offensive language
   - Threats > Offensive language
   - NSFW + Hate speech = Both categories
4. Provide confidence level based on:
   HIGH: Clear match with examples and criteria
   MEDIUM: Matches some criteria but has ambiguous elements
   LOW: Could fit category but uncertain interpretation

FORMAT:
Category: [category_name]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [1-2 sentences explaining specific criteria matched]

Remember: Over-classification is as problematic as under-classification. When in doubt, explain your reasoning.
"""


def moderate_content(text):
    prompt = text
    return client.chat.completions.create(
        model="microsoft/Phi-3.5-mini-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
        temperature=0,
    )


def load_benchmark_data(file_path: str, sample=None):
    df = pd.read_json(file_path, lines=True)
    df["text"] = df["text"].apply(lambda x: x[:2000])
    df = df.sample(frac=1).reset_index(drop=True)

    if sample:
        df = df.head(sample)

    if "text_id" not in df.columns:
        df["text_id"] = range(len(df))

    return df.to_dict("records")


def process_item(item):
    try:
        response = moderate_content(item["text"])
        return {
            "text_id": item["text_id"],
            "text": item["text"],
            "actual_category": item["moderation_category"],
            "model_response": (
                response.choices[0].message.content if response.choices else ""
            ),
            "raw_response": response.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error processing text_id {item.get('text_id', 'unknown')}: {str(e)}")
        return None


def run_benchmark(
    benchmark_data,
    batch_size=16,
    output_dir="benchmark_results",
    model_name="phi35_sglang",
    max_workers=4,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_results_{timestamp}.jsonl")
    results = []
    total_time = 0

    # Open the file once at the start in append mode
    with open(output_file, "a") as f:
        for batch_start in tqdm(range(0, len(benchmark_data), batch_size)):
            batch = benchmark_data[batch_start : batch_start + batch_size]
            batch_start_time = time.time()
            batch_results = []

            # Process batch using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_item = {
                    executor.submit(process_item, item): item for item in batch
                }

                for future in as_completed(future_to_item):
                    result = future.result()
                    if result:
                        batch_results.append(result)
                        results.append(result)

            # Write all results from this batch at once
            for result in batch_results:
                f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure the batch is written to disk

            batch_time = time.time() - batch_start_time
            total_time += batch_time

    print(f"\nBenchmark completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(benchmark_data):.2f} seconds")
    print(f"Results saved to: {output_file}")

    return results


def load_previous_results(file_path: str):
    results = []
    with open(file_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results

def extract_category(model_response):
    """Parse the model response to extract category, with fallback handling."""
    try:
        # Use regex for more robust matching
        category_match = re.search(r"Category:\s*(\w+(?:_?\w+)*)", model_response, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
            # Validate against known categories
            valid_categories = {'hate_or_discrimination', 'violence_or_threats',
                              'offensive_language', 'nsfw_content',
                              'spam_or_scams', 'clean'}
            return category if category in valid_categories else "clean"
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return "error_parsing"  # Default to clean on parsing errors

    return "no_category_found"  # Default if no category found

def create_comparison_file(
    previous_results, new_results, output_file="phi_before_after.jsonl"
):
    comparison_data = []

    # Create a dictionary of new results for easy lookup
    new_results_dict = {r["text_id"]: r for r in new_results}

    for prev_result in previous_results:
        text_id = prev_result["text_id"]
        if text_id in new_results_dict:
            comparison = {
                "text": prev_result["text"],
                "actual_category": prev_result["actual_category"],
                "old_prediction": prev_result["processed_response"][
                    "predicted_category"
                ],
                "new_prediction": extract_category(
                    new_results_dict[text_id]["model_response"]
                ),
            }
            comparison_data.append(comparison)

    # Write comparisons to file
    with open(output_file, "w") as f:
        for item in comparison_data:
            f.write(json.dumps(item) + "\n")




if __name__ == "__main__":
    # Load previous results
    previous_results = load_previous_results(
        "./phi35_benchmark_results_20250210_010356.jsonl"
    )
    previous_results = previous_results[:128]

    # Create benchmark data from previous results
    benchmark_data = [
        {
            "text_id": item["text_id"],
            "text": item["text"],
            "moderation_category": item["actual_category"],
        }
        for item in previous_results
    ]

    # Run benchmark with new prompt
    new_results = run_benchmark(
        benchmark_data=benchmark_data,
        batch_size=16,
        output_dir="benchmark_results",
        model_name="phi35_new_prompt",
    )

    # Create comparison file
    create_comparison_file(previous_results, new_results)
