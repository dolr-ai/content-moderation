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
    base_url="http://localhost:8890/v1",
    api_key=API_KEY,
)

# Defining the system prompt with generalized examples
SYSTEM_PROMPT = """You are a content moderation expert. Your task is to analyze content and categorize it into one of the following categories:

1. hate_or_discrimination : Content that expresses, incites, or promotes hate based on protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status, etc.), including hostile language, prejudiced views, or discriminatory ideology.

Examples of hate_or_discrimination:
- Direct use of slurs or derogatory terms (e.g., "[slur] don't belong here")
- Claims of group inferiority (e.g., "[group] are naturally less intelligent")
- Coded discriminatory language (e.g., "Those people are ruining our neighborhood")
- Identity-based stereotyping (e.g., "All [group] are criminals")
- Dehumanizing comparisons (e.g., comparing protected groups to animals/objects)

2. violence_or_threats : Content that threatens, depicts, or promotes violence, death, or physical harm. This includes graphic violence, threats of serious harm, instructions for violent acts, or glorification of violent behavior.

Examples of violence_or_threats:
- Warnings about physical confrontation (e.g., "If we ever cross paths, you'll regret it")
- Direct threats of lethal violence (e.g., "I will [violent action] you with [weapon]")
- Calls for harm against groups (e.g., "Death to all [group]")
- Implied threats of future violence (e.g., "You better watch your back")
- Specific threats with location references (e.g., "Don't show up at [location] or else")

3. offensive_language : Harassing or severely inappropriate content that attacks, demeans, or abuses others, including hostile expressions, severe profanity, and malicious insults - even when not targeting protected characteristics.

4. nsfw_content : Material intended to arouse sexual excitement, explicit sexual content, or promotion of sexual services (excluding educational/wellness content). Special attention to any content involving minors, which requires immediate escalation.

5. spam_or_scams : Deceptive, fraudulent, or unsolicited content designed to mislead or exploit users, including phishing attempts, scam schemes, coordinated inauthentic behavior, and bulk unsolicited messages.

6. clean : Content that is allowed on platform and does not fall in the categories mentioned above.

For each piece of content, you should:
1. If no violations are found, classify as "clean"
2. Provide your confidence level
3. Provide one or two line explanation

Please format your response exactly as:
Category: [exact category_name]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [short 1/2 line explanation]"""

# Defining the user prompt template
USER_PROMPT = """Analyze this content:
{text}"""


def moderate_content(text):
    return client.chat.completions.create(
        model="microsoft/Phi-3.5-mini-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(text=text)},
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


def load_previous_results(file_path: str, sample=None):
    """Load JSONL file."""
    df = pd.read_json(file_path, lines=True)
    if isinstance(sample, int):
        df = df.sample(n=sample).reset_index(drop=True)

    return df.to_dict("records")


def extract_category(model_response):
    """Parse the model response to extract category, with fallback handling."""
    try:
        # Use regex for more robust matching
        category_match = re.search(
            r"Category:\s*(\w+(?:_?\w+)*)", model_response, re.IGNORECASE
        )
        if category_match:
            category = category_match.group(1).lower()
            # Validate against known categories
            valid_categories = {
                "hate_or_discrimination",
                "violence_or_threats",
                "offensive_language",
                "nsfw_content",
                "spam_or_scams",
                "clean",
            }
            return category if category in valid_categories else "clean"
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return "error_parsing"  # Default to clean on parsing errors

    return "no_category_found"  # Default if no category found


def create_comparison_file(
    previous_results, new_results, output_file="phi_before_after.jsonl"
):
    """Create a comparison file between old and new model results with proper merging."""
    # Convert to dataframes for easier manipulation
    df_new = pd.DataFrame(new_results)
    df_old = pd.DataFrame(previous_results)

    # Extract categories from new results
    df_new["new_prediction"] = df_new["model_response"].apply(extract_category)

    # Extract categories from old results
    df_old["old_prediction"] = df_old["processed_response"].apply(
        lambda x: x["predicted_category"]
    )

    # Merge the dataframes
    df_comparison = df_new[
        ["text_id", "text", "actual_category", "new_prediction"]
    ].merge(df_old[["text_id", "old_prediction"]], on="text_id", how="left")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_file}_{timestamp}.jsonl"
    # Save to jsonl file
    df_comparison.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    # Load previous results
    previous_results = load_previous_results(
        "./phi35_benchmark_results_20250210_010356.jsonl", sample=2048
    )

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
        model_name="phi35_with_new_prompt",
    )

    # Create comparison file
    create_comparison_file(previous_results, new_results)
