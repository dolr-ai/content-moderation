# %%
import os
import openai
from tqdm.auto import tqdm
from typing import Dict, Optional, List
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
    primary_category: str
    secondary_category: Optional[str]
    confidence: str
    cot: str
    raw_response: str

SYSTEM_PROMPT = """You are a content moderation expert. Analyze content by following this step-by-step reasoning process:

Step 1: Pattern Detection
- What are the key phrases and terms?
- Do they match any common violation patterns?
- Are there multiple patterns present?
- How strong are these pattern matches?

Step 2: Context Analysis
- What is the overall tone and intent?
- Is there missing context needed?
- Could this have multiple interpretations?
- What relationship/setting is implied?

Step 3: Violation Assessment
- Are protected characteristics targeted?
- Is there threat or violence?
- Is adult/NSFW content present?
- Is this deceptive or spam-like?
- Is offensive language used?

Step 4: Category Determination
First check if content matches severe violations:
- Targeting protected groups → hate_or_discrimination
- Threats/violence → violence_or_threats
- Adult/sexual content → nsfw_content
- Deceptive patterns → spam_or_scams
Then consider:
- General offensive language
- Clean if no violations

Step 5: Secondary Category Check
- Do additional violations exist?
- Are there supporting patterns?
- Would a second category add value?

Step 6: Confidence Evaluation
- How clear are the pattern matches?
- Is context fully understood?
- Are there conflicting indicators?
- Does more evidence strengthen case?

Output format:
<START_THOUGHT>
Step 1 - Patterns Found:
[Describe key patterns and matches identified]

Step 2 - Context Understanding:
[Explain context and possible interpretations]

Step 3 - Violations Present:
[List potential violations and their strength]

Step 4 - Primary Category Selection:
[Explain category choice and reasoning]

Step 5 - Secondary Category Decision:
[Explain if/why secondary category needed]

Step 6 - Confidence Assessment:
[Justify confidence level assigned]

Classification:
Primary Category: [category]
Secondary Category: [category or None]
Confidence: [HIGH/MEDIUM/LOW]
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
                r"<START_THOUGHT>\s*([\s\S]*?)\s*<END_THOUGHT>",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )

            if thought_match:
                thought_process = thought_match.group(1).strip()
            else:
                print("Warning: No thought process found in response")
                thought_process = "No thought process found"

            # Extract primary category
            primary_cat_match = re.search(
                r"Primary Category:\s*(\w+(?:_?\w+)*)", response_text, re.IGNORECASE
            )
            primary_category = primary_cat_match.group(1).lower() if primary_cat_match else "clean"

            # Extract secondary category
            secondary_cat_match = re.search(
                r"Secondary Category:\s*(\w+(?:_?\w+)*|None)", response_text, re.IGNORECASE
            )
            secondary_category = (
                secondary_cat_match.group(1).lower() if secondary_cat_match else None
            )
            if secondary_category == "none":
                secondary_category = None

            # Validate categories
            if primary_category not in self.valid_categories:
                primary_category = "clean"
            if secondary_category and secondary_category not in self.valid_categories:
                secondary_category = None

            # Extract confidence level
            confidence_match = re.search(
                r"Confidence:\s*(HIGH|MEDIUM|LOW)", response_text, re.IGNORECASE
            )
            confidence = confidence_match.group(1).upper() if confidence_match else "LOW"

            return ModerationResult(
                primary_category=primary_category,
                secondary_category=secondary_category,
                confidence=confidence,
                cot=thought_process,
                raw_response=response_text,
            )

        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            return ModerationResult(
                primary_category="clean",
                secondary_category=None,
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
            "max_tokens": max_tokens,  # Increased for CoT responses
            "temperature": temperature,
        },
    ) as response:
        return await response.json()


async def process_batch_async(batch: List[Dict], session, max_tokens=256, temperature=0):
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
                    "pred_primary_category": moderation_result.primary_category,
                    "pred_secondary_category": moderation_result.secondary_category,
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

    print(f"\nBenchmark completed!")
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

    # Use the correct column names from the new results
    df_new["new_primary"] = df_new["pred_primary_category"]
    df_new["new_secondary"] = df_new["pred_secondary_category"]
    df_new["new_confidence"] = df_new["confidence"]

    # Extract categories from old results - updated to match the actual format
    df_old["old_primary"] = df_old["processed_response"].apply(
        lambda x: x.get("predicted_category", "clean")
    )
    # Old results might not have secondary categories, defaulting to None
    df_old["old_secondary"] = None
    # Extract confidence from scores if available
    df_old["old_confidence"] = df_old["processed_response"].apply(
        lambda x: "HIGH" if x.get("predicted_score", 0) > 0.8
        else "MEDIUM" if x.get("predicted_score", 0) > 0.5
        else "LOW"
    )

    # First merge to get the old results columns
    df_comparison = df_new.merge(
        df_old[["text_id", "old_primary", "old_secondary", "old_confidence"]],
        on="text_id",
        how="left"
    )

    # Select final columns in desired order
    df_comparison = df_comparison[[
        "text_id",
        "text",
        "actual_category",
        "new_primary",
        "new_secondary",
        "new_confidence",
        "old_primary",
        "old_secondary",
        "old_confidence",
        "cot",
    ]]

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
            batch_size=4,
            concurrent_batches=4,
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
