# %%
import os
import openai
from tqdm.auto import tqdm
from typing import Dict, List
import re
from dataclasses import dataclass
import asyncio
import aiohttp
import json
import time
from datetime import datetime
import pandas as pd

API_KEY = "None"


@dataclass
class ModerationResult:
    category: str
    confidence: str
    cot: str
    raw_response: str


SYSTEM_PROMPT = """You are a content moderation expert. Your task is to carefully analyze content by following these steps:

Step 1: Initial Content Review
- Read the content thoroughly
- Identify key terms and phrases
- Note any concerning elements

Step 2: Protected Characteristics Check
- Look for mentions of:
  * Race, gender, ethnicity
  * Religion, nationality
  * Sexual orientation
  * Disability status
- Note if any are mentioned negatively

Step 3: Threat Assessment
- Check for violent language
- Identify direct or implied threats
- Look for location specifics
- Note weapons mentions

Step 4: Language Analysis
- Identify profanity or vulgar terms
- Check for hostile intent
- Look for personal attacks
- Note the overall tone

Step 5: NSFW/Adult Content Check
- Look for explicit sexual content
- Check for suggestive language
- Note any concerning age-related content

Step 6: Spam/Scam Detection
- Look for deceptive language
- Check for unsolicited offers
- Identify suspicious links/codes
- Note unusual formatting or obfuscation

Step 7: Final Classification
Choose ONE of these categories:
1. hate_or_discrimination: Content targeting protected characteristics with negative intent
2. violence_or_threats: Content threatening or promoting violence/harm
3. offensive_language: Hostile content WITHOUT targeting protected characteristics
4. nsfw_content: Explicit sexual content or services
5. spam_or_scams: Deceptive or unsolicited content
6. clean: Content with no violations

Output format (you must follow this exact format):

<START_THOUGHT>
Step 1: [Your initial content review findings]
Step 2: [Your protected characteristics findings]
Step 3: [Your threat assessment findings]
Step 4: [Your language analysis findings]
Step 5: [Your NSFW content findings]
Step 6: [Your spam/scam findings]
Step 7: [Your reasoning for final classification]

Final Classification:
Final Classification:
Category: [category]
Confidence: [HIGH/MEDIUM/LOW]
<END_THOUGHT>
<END_THOUGHT>
"""

USER_PROMPT = """Analyze this content:
{text}
"""
# %%


class ContentModerationCoT:
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8890/v1",
        max_tokens: int = 256,
        temperature: float = 0,
    ):
        self.client = openai.Client(
            base_url=base_url,
            api_key=api_key,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.valid_categories = {
            "hate_or_discrimination",
            "violence_or_threats",
            "offensive_language",
            "nsfw_content",
            "spam_or_scams",
            "clean",
        }

    def parse_response(self, response_text: str) -> ModerationResult:
        """Parse the model's response into structured format with thought process"""
        try:
            # Extract thought process with improved regex
            thought_match = re.search(
                r"<START_THOUGHT>\s*([\s\S]*?)\s*<END_THOUGHT>",  # Changed to [\s\S] to match any character including newlines
                response_text,
                re.IGNORECASE | re.DOTALL,
            )

            if thought_match:
                thought_process = thought_match.group(1).strip()
            else:
                print("Warning: No thought process found in response")
                thought_process = "No thought process found"

            # Extract category and confidence after the thought process
            category_match = re.search(
                r"Category:\s*(\w+(?:_?\w+)*)", response_text, re.IGNORECASE
            )
            category = category_match.group(1).lower() if category_match else "clean"

            # Validate category
            if category not in self.valid_categories:
                category = "clean"

            # Extract confidence level
            confidence_match = re.search(
                r"Confidence:\s*(HIGH|MEDIUM|LOW)", response_text, re.IGNORECASE
            )
            confidence = (
                confidence_match.group(1).upper() if confidence_match else "LOW"
            )

            # Extracted chain of thought
            cot = thought_process if thought_process else "No thought process"

            # Add debug printing
            # print("Raw response:", response_text)
            # print(
            #     "Thought match:",
            #     thought_match.group(0) if thought_match else "No match",
            # )

            return ModerationResult(
                category=category,
                confidence=confidence,
                cot=cot,
                raw_response=response_text,
            )

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return ModerationResult(
                category="clean",
                confidence="LOW",
                cot="Error parsing response",
                raw_response=response_text,
            )

    def moderate_content(self, text: str) -> ModerationResult:
        """Moderate content using the model with chain of thought prompting"""

        response = self.client.chat.completions.create(
            model="microsoft/Phi-3.5-mini-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(text=text)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        response_text = response.choices[0].message.content
        return self.parse_response(response_text)


# %%


async def moderate_content_async(session, text, max_tokens=256, temperature=0):
    """Async version of content moderation that returns raw API response"""
    async with session.post(
        "http://localhost:8890/v1/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model": "microsoft/Phi-3.5-mini-instruct",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(text=text)},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    ) as response:
        return await response.json()


async def process_batch_async(
    batch: List[Dict], session, max_tokens=256, temperature=0
):
    """Process a batch of content using the CoT approach"""
    tasks = []
    for item in batch:
        task = asyncio.create_task(
            moderate_content_async(
                session, item["text"], max_tokens=max_tokens, temperature=temperature
            )
        )
        tasks.append((item, task))

    # Create a single moderator instance for parsing responses
    cot_moderator = ContentModerationCoT(api_key="None")

    results = []
    for item, task in tasks:
        try:
            response = await task
            response_text = (
                response["choices"][0]["message"]["content"]
                if response.get("choices")
                else ""
            )

            # Use the CoT moderator to parse the response
            moderation_result = cot_moderator.parse_response(response_text)

            results.append(
                {
                    "text_id": item["text_id"],
                    "text": item["text"],
                    "actual_category": item["moderation_category"],
                    "model_response": response_text,
                    "predicted_category": moderation_result.category,
                    "confidence": moderation_result.confidence,
                    "cot": moderation_result.cot,
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
    model_name="phi35_cot",
    concurrent_batches=4,
    timestamp=None,
    max_tokens=256,
    temperature=0,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
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
            tasks = [
                process_batch_async(
                    batch, session, max_tokens=max_tokens, temperature=temperature
                )
                for batch in batches
            ]
            batch_results = await asyncio.gather(*tasks)

            with open(output_file, "a") as f:
                for results in batch_results:
                    for result in results:
                        f.write(json.dumps(result) + "\n")
                        all_results.append(result)

            chunk_time = time.time() - chunk_start_time
            total_time += chunk_time
            await asyncio.sleep(0.1)

    print("\nBenchmark completed!")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time/len(benchmark_data):.2f} seconds")
    print(f"Results saved to: {output_file}")

    return all_results, timestamp


def create_comparison_file(
    previous_results,
    new_results,
    output_dir="benchmark_results",
    timestamp=None,
    output_prefix="phi_cot_comparison",
):
    df_new = pd.DataFrame(new_results)
    df_old = pd.DataFrame(previous_results)

    # No need to create new moderator instance since predictions are already parsed
    df_new["new_prediction"] = df_new["predicted_category"]
    df_old["old_prediction"] = df_old["processed_response"].apply(
        lambda x: x["predicted_category"]
    )

    # Merge the dataframes
    df_comparison = df_new[
        ["text_id", "text", "actual_category", "new_prediction", "cot"]
    ].merge(df_old[["text_id", "old_prediction"]], on="text_id", how="left")

    output_file = os.path.join(output_dir, f"{output_prefix}_{timestamp}.jsonl")
    df_comparison.to_json(output_file, orient="records", lines=True)
    print(f"Saved comparison file to {output_file}")


def load_previous_results(file_path: str, sample=None):
    """Load JSONL file."""
    df = pd.read_json(file_path, lines=True)
    if isinstance(sample, int):
        df = df.sample(n=sample).reset_index(drop=True)

    return df.to_dict("records")


if __name__ == "__main__":
    output_dir = "benchmark_results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    max_tokens = 512
    temperature = 0

    # Load previous results
    previous_results = load_previous_results(
        "./phi35_benchmark_results_20250210_010356.jsonl",
        sample=2048,
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

    # Run benchmark with CoT prompt using async version
    new_results, _ = asyncio.run(
        run_benchmark_async(
            benchmark_data=benchmark_data,
            batch_size=16,
            concurrent_batches=2,
            output_dir=output_dir,
            model_name="phi35_with_cot",
            timestamp=timestamp,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    )

    # Create comparison file
    create_comparison_file(
        previous_results, new_results, output_dir=output_dir, timestamp=timestamp
    )
