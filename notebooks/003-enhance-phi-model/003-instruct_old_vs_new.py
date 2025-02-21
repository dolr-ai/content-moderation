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

# Defining the system prompt with generalized examples
SYSTEM_PROMPT = """You are a content moderation expert. Your task is to analyze content and categorize it into one of the following categories:

1. hate_or_discrimination: Content targeting protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status). Must include BOTH:
- A protected characteristic AND
- Negative intent/prejudice towards that characteristic

Examples (ALL must target protected characteristics):
- Slurs of any kind, in any context
- Group stereotyping ("All [group] are...")
- Dehumanizing comparisons
- Coded bigotry ("Those people", "(((they)))")
- Claims of inferiority
- "Jokes" targeting identity

NOT hate_or_discrimination (belongs in offensive_language):
- General insults ("idiot", "stupid")
- Non-identity profanity
- Personal attacks without targeting identity
- Vulgar language without prejudice
- Angry/hostile comments without targeting protected groups


2. violence_or_threats : Content that threatens, depicts, or promotes violence, death, or physical harm. This includes graphic violence, threats of serious harm, instructions for violent acts, or glorification of violent behavior.

Examples of violence_or_threats:
- Warnings about physical confrontation (e.g., "If we ever cross paths, you'll regret it")
- Direct threats of lethal violence (e.g., "I will [violent action] you with [weapon]")
- Calls for harm against groups (e.g., "Death to all [group]")
- Implied threats of future violence (e.g., "You better watch your back")
- Specific threats with location references (e.g., "Don't show up at [location] or else")

3. offensive_language: Hostile or inappropriate content WITHOUT targeting protected characteristics:
- Personal insults
- Profanity
- Vulgar remarks
- Name-calling
- Aggressive hostility

Precedence Rule:
- If content includes BOTH offensive language AND references to protected characteristics, classify as hate_or_discrimination
- When in doubt about protected characteristics, classify as offensive_language

Examples of offensive_language:
- Direct personal attacks with profanity (e.g., "You're a worthless piece of [profanity]")
- Aggressive insults about intelligence/appearance (e.g., "You stupid [insult], can't even read")
- Vulgar sexual remarks not rising to NSFW level (e.g., "Go [vulgar act] yourself")
- Repeated hostile name-calling (e.g., "You're nothing but a [insult] [insult] [insult]")
- Non-English vulgar terms with clear hostile intent (e.g., "[non-English insult], you're pathetic")

NOT offensive_language:
- Mild profanity without hostile intent
- Casual swearing in friendly context
- General disagreement without personal attacks
- Clinical/medical terms
- Creative criticism without vulgar language

4. nsfw_content : Material intended to arouse sexual excitement, explicit sexual content, or promotion of sexual services (excluding educational/wellness content). Special attention to any content involving minors, which requires immediate escalation.

5. spam_or_scams : Deceptive, fraudulent, or unsolicited content designed to mislead or exploit users, including phishing attempts, scam schemes, coordinated inauthentic behavior, and bulk unsolicited messages.

Examples of spam_or_scams:
- Unsolicited product promotions (e.g., "Buy genuine R0LEX watches at 90% discount!")
- Medication/pharmaceutical spam (e.g., "Get cheap med!cations without prescription")
- Account verification phishing (e.g., "Your account requires urgent verification. Click here: [link]")
- Prize/reward scams (e.g., "Congratulations! Claim your free [product] now. Code: XXX")
- Encoded/obfuscated spam (e.g., "V1agra" or "Med!cation" with intentionally altered spellings)

NOT spam_or_scams:
- Legitimate promotional codes or offers from verified sources
- Regular business communications or newsletters
- Automated system notifications
- Customer service messages
- Standard email forwards without deceptive content

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
    previous_results, new_results, output_file="phi_before_after"
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
