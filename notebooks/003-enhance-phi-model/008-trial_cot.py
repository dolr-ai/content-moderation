# %%
import os
import openai
from typing import Dict, Optional, List
import re
from dataclasses import dataclass


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
<END_THOUGHT>

Based on my step-by-step analysis:
Category: [category]
Confidence: [HIGH/MEDIUM/LOW]
"""

USER_PROMPT = """Analyze this content:
{text}
"""
# %%


class ContentModerationCoT:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8890/v1"):
        self.client = openai.Client(
            base_url=base_url,
            api_key=api_key,
        )
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

            print(f"Thought process: {thought_process}")
            # Extracted chain of thought
            cot = thought_process if thought_process else "No thought process"

            # Add debug printing
            print("Raw response:", response_text)
            print(
                "Thought match:",
                thought_match.group(0) if thought_match else "No match",
            )

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
            temperature=0,
            max_tokens=512,
        )

        response_text = response.choices[0].message.content
        return self.parse_response(response_text)


# Example usage
# %%
API_KEY = "None"
moderator = ContentModerationCoT(API_KEY)
# %%
test_cases = [
    {
        "text": """What exactly does saying ""How are you niggers today?"" have to do with this discussion?  ""Nigger"" is a far more inflamatory word than ""homo"" is!  There are far more derogatory terms for homosexuals than homos - that would be on par with ""nigger"" - but I did not use any of them.  And whether admins are ""stupid"" or not is equally irrelevant.""",
        "correct_category": "hate_or_discrimination",
        "common_wrong_category": "clean",
    },
]
for d in test_cases:
    print(f"\nAnalyzing: {d['text']}")
    result = moderator.moderate_content(d["text"])
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence}")
    print(f"Chain of Thought: {result.cot}")
    print("-" * 50)
    break
# %%
# test_cases[-1:]
print(result.cot)
