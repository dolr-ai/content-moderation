# %%
import re
import pprint
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union

DEV_CONFIG_PATH = "/Users/sagar/work/yral/content-moderation/dev_config.yml"

with open(DEV_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(config["local"]["PROJECT_ROOT"])
DATA_ROOT = Path(config["local"]["DATA_ROOT"])

# %% [markdown]
# ## Check Llama32-1B and Llama32-3B Parsing

# %%
check_model = "llama32-3"

if check_model == "llama32-1B":
    df = pd.read_json(
        DATA_ROOT
        / "benchmark_results/llm/llama32-1B_benchmark_results_20250207_083439.jsonl",
        lines=True,
    )
elif check_model == "llama32-3B":
    df = pd.read_json(
        DATA_ROOT
        / "benchmark_results/llm/llama32-3B_benchmark_results_20250210_064258.jsonl",
        lines=True,
    )

# %%
CATEGORIES = {
    "hate_or_discrimination": "Content that expresses, incites, or promotes hate based on protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status, etc.), including hostile language, prejudiced views, or discriminatory ideology.",
    "violence_or_threats": "Content that threatens, depicts, or promotes violence, death, or physical harm. This includes graphic violence, threats of serious harm, instructions for violent acts, or glorification of violent behavior.",
    "offensive_language": "Harassing or severely inappropriate content that attacks, demeans, or abuses others, including hostile expressions, severe profanity, and malicious insults - even when not targeting protected characteristics.",
    "nsfw_content": "Material intended to arouse sexual excitement, explicit sexual content, or promotion of sexual services (excluding educational/wellness content). Special attention to any content involving minors, which requires immediate escalation.",
    "spam_or_scams": "Deceptive, fraudulent, or unsolicited content designed to mislead or exploit users, including phishing attempts, scam schemes, coordinated inauthentic behavior, and bulk unsolicited messages.",
    "clean": "Content that is allowed on platform and does not fall in the categories mentioned above.",
}


def parse_response(response: str) -> Dict[str, Union[str, float]]:
    """Parse the generated response to extract category and confidence."""
    category_match = re.search(r"Category:\s*(\w+(?:_?\w+)*)", response, re.IGNORECASE)
    confidence_match = re.search(
        r"Confidence:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE
    )

    category = category_match.group(1).lower() if category_match else "clean"
    confidence = confidence_match.group(1).upper() if confidence_match else "LOW"

    if category not in CATEGORIES:
        print(f"Warning: Invalid category '{category}' detected, defaulting to 'clean'")
        category = "clean"

    confidence_scores = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    score = confidence_scores.get(confidence, 0.3)

    return {
        "category": category,
        "confidence": score,
        # edited for better readability
        # actual benchmark response has both the parts
        "raw_response": response.split("<|start_header_id|>assistant<|end_header_id|>")[
            1
        ].replace("<|eot_id|>", ""),
    }


df["parsed_response"] = df["raw_llm_response"].apply(parse_response)
df["predicted_category"] = df["parsed_response"].apply(lambda x: x["category"])
df["explanation"] = df["parsed_response"].apply(lambda x: x["raw_response"])


# %%
random_ix = np.random.randint(0, len(df))
txt = df["text"].iloc[random_ix]
llm_response = df["raw_llm_response"].iloc[random_ix]
actual_category = df["actual_category"].iloc[random_ix]
print("Text: \n", txt)
print("Actual Category: \n", actual_category)
print("Parsed Response: \n", pprint.pformat(df["parsed_response"].iloc[random_ix]))

# %%[markdown]
# ## Check Llamaguard Parsing

check_model = "llamaguard-8B"

if check_model == "llamaguard-1B":
    df = pd.read_json(
        DATA_ROOT
        / "benchmark_results/llm/llamaguard-1B_benchmark_results_20250210_014647.jsonl",
        lines=True,
    )

elif check_model == "llamaguard-8B":
    df = pd.read_json(
        DATA_ROOT
        / "benchmark_results/llm/llamaguard-8B_benchmark_results_20250210_090137.jsonl",
        lines=True,
    )


# %%
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


def parse_response(response: str) -> Dict[str, Union[str, float]]:
    """Parse the generated response to extract safety assessment and categories."""
    # Extract the part after the last assistant header
    parts = response.split("<|start_header_id|>assistant<|end_header_id|>")
    if len(parts) > 1:
        response_text = parts[-1].strip()
    else:
        response_text = response.strip()

    # Clean up response text
    response_text = response_text.replace("<|eot_id|>", "").strip()

    # Split into lines and clean up
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    # Initialize variables
    is_safe = True  # default to safe
    violated_codes = []

    # Analyze the response line by line
    for line in lines:
        line = line.lower().strip()
        if line == "unsafe":
            is_safe = False
        elif not is_safe:  # Only look for categories if marked as unsafe
            # Look for category codes in the current line
            codes = re.findall(r"S\d+", line.upper())
            if codes:
                violated_codes.extend(codes)

    # Get unique category codes
    violated_codes = list(dict.fromkeys(violated_codes))

    # Map violated categories
    violated_categories = [
        CATEGORIES.get(code) for code in violated_codes if code in CATEGORIES
    ]

    # Determine platform category
    platform_category = "clean"
    if violated_codes:
        first_code = violated_codes[0]
        platform_category = LLAMAGUARD_TO_PLATFORM_CATEGORY_MAPPING.get(
            first_code, "clean"
        )

    return {
        "category": platform_category,
        "violated_llamaguard_categories": violated_categories,
        "confidence": 0.9 if is_safe else 0.8,
        "raw_response": response.replace("<|eot_id|>", "").split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[1],
    }


df["parsed_response"] = df["raw_llm_response"].apply(parse_response)
df["predicted_category"] = df["parsed_response"].apply(lambda x: x["category"])
df["explanation"] = df["parsed_response"].apply(lambda x: x["raw_response"])


# %%
random_ix = np.random.randint(0, len(df))
txt = df["text"].iloc[random_ix]
llm_response = df["raw_llm_response"].iloc[random_ix]
actual_category = df["actual_category"].iloc[random_ix]
print("Text: \n", txt[:2000])
print("Actual Category: \n", actual_category)
print("Parsed Response: \n", pprint.pformat(df["parsed_response"].iloc[random_ix]))


# %%

df["safe_predicted"] = df["parsed_response"].apply(
    lambda x: len(x["violated_llamaguard_categories"]) == 0
)

df["safe_predicted"].value_counts()

df[df["actual_category"] == "clean"].shape

df["safe_actual"] = df["actual_category"] == "clean"

df["safe_predicted"] = df["safe_predicted"] & (df["predicted_category"] == "clean")


# %%
confusion_matrix = pd.crosstab(df["safe_actual"], df["safe_predicted"])
print(confusion_matrix.to_string())

# %%[markdown]
# ## Check phi35 Parsing

check_model = "phi35"

if check_model == "phi35":
    df = pd.read_json(
        DATA_ROOT
        / "benchmark_results/llm/phi35_benchmark_results_20250210_010356.jsonl",
        lines=True,
    )

# %%

CATEGORIES = {
    "hate_or_discrimination": "Content that expresses, incites, or promotes hate based on protected characteristics (race, gender, ethnicity, religion, nationality, sexual orientation, disability status, etc.), including hostile language, prejudiced views, or discriminatory ideology.",
    "violence_or_threats": "Content that threatens, depicts, or promotes violence, death, or physical harm. This includes graphic violence, threats of serious harm, instructions for violent acts, or glorification of violent behavior.",
    "offensive_language": "Harassing or severely inappropriate content that attacks, demeans, or abuses others, including hostile expressions, severe profanity, and malicious insults - even when not targeting protected characteristics.",
    "nsfw_content": "Material intended to arouse sexual excitement, explicit sexual content, or promotion of sexual services (excluding educational/wellness content). Special attention to any content involving minors, which requires immediate escalation.",
    "spam_or_scams": "Deceptive, fraudulent, or unsolicited content designed to mislead or exploit users, including phishing attempts, scam schemes, coordinated inauthentic behavior, and bulk unsolicited messages.",
    "clean": "Content that is allowed on platform and does not fall in the categories mentioned above.",
}


def parse_response(response: str) -> Dict[str, Union[str, float]]:
    """Parse the generated response to extract category and confidence."""
    category_match = re.search(r"Category:\s*(\w+(?:_?\w+)*)", response, re.IGNORECASE)
    confidence_match = re.search(
        r"Confidence:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE
    )

    category = category_match.group(1).lower() if category_match else "clean"
    confidence = confidence_match.group(1).upper() if confidence_match else "LOW"

    if category not in CATEGORIES:
        print(f"Warning: Invalid category '{category}' detected, defaulting to 'clean'")
        category = "clean"

    confidence_scores = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
    score = confidence_scores.get(confidence, 0.3)

    return {
        "category": category,
        "confidence": score,
        "raw_response": response.split("<|end|><|user|>")[1],
    }


df["parsed_response"] = df["raw_llm_response"].apply(parse_response)
df["predicted_category"] = df["parsed_response"].apply(lambda x: x["category"])
df["explanation"] = df["parsed_response"].apply(lambda x: x["raw_response"])

# %%
random_ix = np.random.randint(0, len(df))
txt = df["text"].iloc[random_ix]
llm_response = df["raw_llm_response"].iloc[random_ix]
actual_category = df["actual_category"].iloc[random_ix]
print("Text: \n", txt[:2000])
print("Actual Category: \n", actual_category)
print("Parsed Response: \n", pprint.pformat(df["parsed_response"].iloc[random_ix]))
