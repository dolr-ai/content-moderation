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
import asyncio
import aiohttp
from typing import List, Dict

API_KEY = "None"

client = openai.Client(
    base_url="http://localhost:8890/v1",
    api_key=API_KEY,
)

# Content moderation system with primary and secondary category prediction
SYSTEM_PROMPT = """You are a content moderation expert. Analyze content and categorize it into primary and secondary categories:

1. hate_or_discrimination: Content targeting protected characteristics that includes:
   - Protected characteristics: race, gender, ethnicity, religion, nationality, sexual orientation, disability
   - Forms of discrimination:
     * Direct slurs or derogatory terms (even in "friendly" context)
     * Stereotyping ("All [group] are...")
     * Coded bigotry ("Those people", "(((they)))")
     * Claims of group inferiority
     * Dehumanizing comparisons (comparing groups to animals/objects)
     * "Ironic" or "joking" discrimination
     * Anti-identity insults using characteristics as attacks

2. violence_or_threats: Content involving violence or harm:
   - Direct physical threats ("I will [violent action] you")
   - Implied threats ("You better watch your back")
   - Specific violent plans ("Let's get them at [location]")
   - Calls for violence ("Someone needs to [violent act]")
   - Glorification of violence ("Love watching them get hurt")
   - Instructions for harmful acts
   - Threats combined with hate speech (classified as both)

3. offensive_language: Hostile or inappropriate content without targeting protected characteristics:
   - Severe personal attacks ("You're worthless")
   - Aggressive profanity with hostile intent
   - Malicious insults about intelligence/appearance
   - Vulgar sexual remarks (not rising to NSFW level)
   - Repeated hostile name-calling
   - NOT included: mild swearing, friendly banter, clinical terms

4. nsfw_content: Sexual or adult content:
   - Explicit sexual descriptions
   - Pornographic content
   - Sexual solicitation
   - Adult services promotion
   - Sexualization of minors (requires immediate escalation)
   - NOT included: medical/educational content, mild innuendo

5. spam_or_scams: Deceptive or unsolicited content:
   - Phishing attempts ("Verify your account: [suspicious link]")
   - Fake promotions ("90% off luxury items!")
   - Cryptocurrency/investment scams
   - Obfuscated spam ("V1agra", "Med!cation")
   - Unsolicited bulk messaging
   - Coordinated inauthentic behavior
   - NOT included: legitimate promotions, customer service messages

6. clean: Content that:
   - Contains no policy violations
   - May include mild swearing without hostility
   - May include legitimate criticism without attacks
   - May include educational discussion of sensitive topics
   - May include legitimate commercial content

Key Rules:
- If content includes BOTH offensive language AND protected characteristics, classify as hate_or_discrimination
- Primary category should be the most severe violation observed
- Secondary category should be the second most severe violation (if any)
- If no violations, primary = clean, secondary = None

Please format response exactly as:
Primary Category: [category_name]
Secondary Category: [category_name or None]
Confidence: [HIGH/MEDIUM/LOW]
Explanation: [1-2 line explanation]"""

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


def load_previous_results(file_path: str, sample=None):
    """Load JSONL file."""
    df = pd.read_json(file_path, lines=True)
    if isinstance(sample, int):
        df = df.sample(n=sample).reset_index(drop=True)

    return df.to_dict("records")


def extract_categories(model_response):
    """Parse the model response to extract primary and secondary categories, with fallback handling."""
    try:
        # Use regex for more robust matching of both categories
        primary_match = re.search(
            r"Primary Category:\s*(\w+(?:_?\w+)*)", model_response, re.IGNORECASE
        )
        secondary_match = re.search(
            r"Secondary Category:\s*(\w+(?:_?\w+)*|None)", model_response, re.IGNORECASE
        )
        confidence_match = re.search(
            r"Confidence:\s*(HIGH|MEDIUM|LOW)", model_response, re.IGNORECASE
        )

        valid_categories = {
            "hate_or_discrimination",
            "violence_or_threats",
            "offensive_language",
            "nsfw_content",
            "spam_or_scams",
            "clean",
        }

        # Extract and validate primary category
        primary = (
            primary_match.group(1).lower() if primary_match else "no_category_found"
        )
        primary = primary if primary in valid_categories else "clean"

        # Extract and validate secondary category
        secondary = secondary_match.group(1).lower() if secondary_match else None
        if secondary and secondary != "none":
            secondary = secondary if secondary in valid_categories else None

        # Extract confidence
        confidence = confidence_match.group(1).upper() if confidence_match else "LOW"

        return {
            "primary_category": primary,
            "secondary_category": secondary,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        return {
            "primary_category": "error_parsing",
            "secondary_category": None,
            "confidence": "LOW",
        }


def create_comparison_file(
    previous_results, new_results, output_file="phi_before_after"
):
    """Create a comparison file between old and new model results with proper merging."""
    # Convert to dataframes for easier manipulation
    df_new = pd.DataFrame(new_results)
    df_old = pd.DataFrame(previous_results)

    # Extract categories from new results
    df_new["new_predictions"] = df_new["model_response"].apply(extract_categories)
    df_new["new_primary"] = df_new["new_predictions"].apply(
        lambda x: x["primary_category"]
    )
    df_new["new_secondary"] = df_new["new_predictions"].apply(
        lambda x: x["secondary_category"]
    )
    df_new["confidence"] = df_new["new_predictions"].apply(lambda x: x["confidence"])

    # Extract categories from old results
    df_old["old_prediction"] = df_old["processed_response"].apply(
        lambda x: x["predicted_category"]
    )

    # Merge the dataframes
    df_comparison = df_new[
        [
            "text_id",
            "text",
            "actual_category",
            "new_primary",
            "new_secondary",
            "confidence",
        ]
    ].merge(df_old[["text_id", "old_prediction"]], on="text_id", how="left")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_file}_{timestamp}.jsonl"
    # Save to jsonl file
    df_comparison.to_json(output_file, orient="records", lines=True)
    print(f"Saved before after file to {output_file}")


async def moderate_content_async(session, text):
    async with session.post(
        "http://localhost:8890/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "microsoft/Phi-3.5-mini-instruct",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(text=text)},
            ],
            "max_tokens": 100,
            "temperature": 0,
        },
    ) as response:
        return await response.json()


async def process_batch_async(batch: List[Dict], session):
    tasks = []
    for item in batch:
        task = asyncio.create_task(moderate_content_async(session, item["text"]))
        tasks.append((item, task))

    results = []
    for item, task in tasks:
        try:
            response = await task
            results.append(
                {
                    "text_id": item["text_id"],
                    "text": item["text"],
                    "actual_category": item["moderation_category"],
                    "model_response": (
                        response["choices"][0]["message"]["content"]
                        if response.get("choices")
                        else ""
                    ),
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            print(
                f"Error processing text_id {item.get('text_id', 'unknown')}: {str(e)}"
            )

    return results


async def run_benchmark_async(
    benchmark_data,
    batch_size=16,
    output_dir="benchmark_results",
    model_name="phi35_sglang",
    concurrent_batches=4,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"{model_name}_results_{timestamp}.jsonl")

    total_time = 0
    all_results = []
    chunk_size = batch_size * concurrent_batches

    async with aiohttp.ClientSession() as session:
        for chunk_start in tqdm(range(0, len(benchmark_data), chunk_size)):
            chunk = benchmark_data[chunk_start : chunk_start + chunk_size]
            chunk_start_time = time.time()

            batches = [
                chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)
            ]
            tasks = [process_batch_async(batch, session) for batch in batches]
            batch_results = await asyncio.gather(*tasks)

            with open(output_file, "a") as f:
                for results in batch_results:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                        all_results.append(result)

            chunk_time = time.time() - chunk_start_time
            total_time += chunk_time
            await asyncio.sleep(0.1)

    print(f"\nBenchmark completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(benchmark_data):.2f} seconds")
    print(f"Results saved to: {output_file}")

    return all_results


if __name__ == "__main__":
    # Load previous results
    previous_results = load_previous_results(
        "./phi35_benchmark_results_20250210_010356.jsonl", sample=4096
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

    # Run benchmark with new prompt using async version
    new_results = asyncio.run(
        run_benchmark_async(
            benchmark_data=benchmark_data,
            batch_size=16,
            concurrent_batches=4,
            # This will allow up to batch_size * concurrent_batches concurrent requests
            output_dir="benchmark_results",
            model_name="phi35_with_new_prompt",
        )
    )

    # Create comparison file
    create_comparison_file(previous_results, new_results)
