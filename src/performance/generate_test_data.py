#!/usr/bin/env python3
"""
Generate test data for performance testing

This script generates a JSONL file with sample texts for performance testing.
"""
import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

# Sample text categories and examples
SAMPLE_TEXTS = {
    "clean": [
        "This is a normal conversation about technology.",
        "I'm interested in learning more about artificial intelligence.",
        "The weather today is quite pleasant.",
        "I enjoyed reading that book about history.",
        "Let's discuss the latest scientific discoveries.",
        "What are your thoughts on renewable energy?",
        "I'm planning to visit the museum this weekend.",
        "The conference was very informative and well-organized.",
        "Could you recommend a good restaurant in the area?",
        "I've been working on improving my programming skills.",
    ],
    "offensive_language": [
        "This product is absolute garbage and a waste of money.",
        "The service at this place is terrible and the staff is incompetent.",
        "That politician is a complete idiot who doesn't understand basic facts.",
        "Your opinion on this matter is totally worthless.",
        "This company's customer support is a joke and they don't care about users.",
    ],
    "hate_or_discrimination": [
        "People from that country are all lazy and dishonest.",
        "Women are not capable of understanding complex technical subjects.",
        "That religious group is responsible for all the problems in society.",
        "People with that disability are a burden on the system.",
        "Immigrants are taking all our jobs and ruining the economy.",
    ],
    "spam_or_scams": [
        "Congratulations! You've won $10,000,000 in our lottery. Click here to claim your prize!",
        "Make $5000 a week working from home with this simple trick!",
        "Limited time offer: Buy one get three free! Act now before supplies run out!",
        "Your account has been compromised. Please verify your details by clicking this link.",
        "Invest now in this cryptocurrency and triple your money in just one week!",
    ],
}


def generate_random_text(min_words: int = 5, max_words: int = 20) -> str:
    """Generate a random text with a specified number of words"""
    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
        "even", "new", "want", "because", "any", "these", "give", "day", "most", "us",
    ]

    num_words = random.randint(min_words, max_words)
    return " ".join(random.choice(words) for _ in range(num_words))


def generate_test_data(
    num_samples: int = 1000,
    output_file: str = "test_data.jsonl",
    category_distribution: Optional[Dict[str, float]] = None,
    include_random: bool = True,
    min_random_words: int = 5,
    max_random_words: int = 50,
) -> None:
    """
    Generate test data for performance testing

    Args:
        num_samples: Number of samples to generate
        output_file: Output JSONL file path
        category_distribution: Distribution of categories (default: equal distribution)
        include_random: Whether to include random texts
        min_random_words: Minimum number of words for random texts
        max_random_words: Maximum number of words for random texts
    """
    # Set default category distribution if not provided
    if category_distribution is None:
        categories = list(SAMPLE_TEXTS.keys())
        if include_random:
            categories.append("random")

        # Equal distribution
        category_distribution = {category: 1.0 / len(categories) for category in categories}

    # Calculate number of samples per category
    samples_per_category = {
        category: int(num_samples * ratio)
        for category, ratio in category_distribution.items()
    }

    # Adjust for rounding errors
    total = sum(samples_per_category.values())
    if total < num_samples:
        # Add remaining samples to the first category
        first_category = list(samples_per_category.keys())[0]
        samples_per_category[first_category] += num_samples - total

    # Generate samples
    samples = []

    for category, count in samples_per_category.items():
        if category == "random":
            # Generate random texts
            for _ in range(count):
                text = generate_random_text(min_random_words, max_random_words)
                samples.append({"text": text})
        else:
            # Use predefined examples and repeat as needed
            examples = SAMPLE_TEXTS[category]
            for i in range(count):
                text = examples[i % len(examples)]
                samples.append({"text": text, "moderation_category": category})

    # Shuffle samples
    random.shuffle(samples)

    # Write to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Generated {len(samples)} test samples and saved to {output_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate test data for performance testing")
    parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--output-file", type=str, default="test_data.jsonl", help="Output JSONL file path"
    )
    parser.add_argument(
        "--clean-ratio", type=float, default=0.4, help="Ratio of clean texts"
    )
    parser.add_argument(
        "--offensive-ratio", type=float, default=0.2, help="Ratio of offensive language texts"
    )
    parser.add_argument(
        "--hate-ratio", type=float, default=0.1, help="Ratio of hate or discrimination texts"
    )
    parser.add_argument(
        "--spam-ratio", type=float, default=0.1, help="Ratio of spam or scam texts"
    )
    parser.add_argument(
        "--random-ratio", type=float, default=0.2, help="Ratio of random texts"
    )
    parser.add_argument(
        "--no-random", action="store_true", help="Don't include random texts"
    )
    parser.add_argument(
        "--min-random-words", type=int, default=5, help="Minimum number of words for random texts"
    )
    parser.add_argument(
        "--max-random-words", type=int, default=50, help="Maximum number of words for random texts"
    )

    args = parser.parse_args()

    # Validate ratios
    ratios = {
        "clean": args.clean_ratio,
        "offensive_language": args.offensive_ratio,
        "hate_or_discrimination": args.hate_ratio,
        "spam_or_scams": args.spam_ratio,
    }

    if not args.no_random:
        ratios["random"] = args.random_ratio

    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Category ratios sum to {total_ratio}, not 1.0")
        # Normalize ratios
        ratios = {category: ratio / total_ratio for category, ratio in ratios.items()}

    # Generate test data
    generate_test_data(
        num_samples=args.num_samples,
        output_file=args.output_file,
        category_distribution=ratios,
        include_random=not args.no_random,
        min_random_words=args.min_random_words,
        max_random_words=args.max_random_words,
    )


if __name__ == "__main__":
    main()