# %%
import os
import requests
import openai
from sglang.utils import print_highlight
import json
import subprocess
import sglang as sgl

API_KEY = "<add-your-api-key>"

client = openai.Client(
    base_url="http://<add-your-internal-ip>:8890/v1",
    api_key=API_KEY,
)


# %%
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

# %%

examples = [
    {
        "text": "Important notice: Your tax refund requires urgent attention. Act as soon as possible to avoid issues. Ref: 8301",
        "correct_category": "spam_or_scams",
        "common_wrong_category": "hate_or_discrimination",
    },
    {
        "text": "Patient presented with swelling in the breast tissue and requires immediate examination.",
        "correct_category": "clean",
        "common_wrong_category": "nsfw_content",
    },
    {
        "text": "Going to kill all the zombies in the new game level! Who wants to join?",
        "correct_category": "clean",
        "common_wrong_category": "violence_or_threats",
    },
    {
        "text": "These immigration policies are destroying our economy. We need stricter controls.",
        "correct_category": "clean",
        "common_wrong_category": "hate_or_discrimination",
    },
    {
        "text": "As a queer person, I'm proud of our community's achievements.",
        "correct_category": "clean",
        "common_wrong_category": "hate_or_discrimination",
    },
    {
        "text": "This is absolute bullshit! Your customer service is the worst I've ever seen!",
        "correct_category": "offensive_language",
        "common_wrong_category": "hate_or_discrimination",
    },
    {
        "text": "Don't miss out! 50% off all items this weekend at our store. Limited time offer!",
        "correct_category": "clean",
        "common_wrong_category": "spam_or_scams",
    },
    {
        "text": "The documentary about World War II showed devastating footage of the battles.",
        "correct_category": "clean",
        "common_wrong_category": "violence_or_threats",
    },
    {
        "text": "BUY V1AGRA N0W!!! Best PR1CES!!! Click here>>>",
        "correct_category": "spam_or_scams",
        "common_wrong_category": "nsfw_content",
    },
    {
        "text": "Let's have a discussion about reproductive health and contraception options.",
        "correct_category": "clean",
        "common_wrong_category": "nsfw_content",
    },
    {
        "text": "I will seriously punch someone if my package doesn't arrive soon!",
        "correct_category": "offensive_language",
        "common_wrong_category": "violence_or_threats",
    },
    {
        "text": "MAKE $5000 DAILY FROM HOME!!! 100% GUARANTEED!!!",
        "correct_category": "spam_or_scams",
        "common_wrong_category": "clean",
    },
    {
        "text": "Your account has been compromised. Click here immediately to verify your identity.",
        "correct_category": "spam_or_scams",
        "common_wrong_category": "clean",
    },
    {
        "text": "I'm tired of these foreigners taking our jobs!",
        "correct_category": "hate_or_discrimination",
        "common_wrong_category": "clean",
    },
    {
        "text": "The breast cancer awareness campaign starts next month.",
        "correct_category": "clean",
        "common_wrong_category": "nsfw_content",
    },
    {
        "text": "I hate this weather! It's driving me crazy!",
        "correct_category": "clean",
        "common_wrong_category": "offensive_language",
    },
    {
        "text": "Your password will expire in 24 hours. Please click here to update it.",
        "correct_category": "spam_or_scams",
        "common_wrong_category": "clean",
    },
    {
        "text": "The sex education curriculum needs to be updated for modern teenagers.",
        "correct_category": "clean",
        "common_wrong_category": "nsfw_content",
    },
    {
        "text": "My fucking car won't start again!",
        "correct_category": "offensive_language",
        "common_wrong_category": "clean",
    },
    {
        "text": "If you don't stop messaging me, I'll report you to the police.",
        "correct_category": "clean",
        "common_wrong_category": "violence_or_threats",
    },
]
# %%
for example in examples:
    print(f"Example: {example['text']}")
    print(f"Correct category: {example['correct_category']}")
    print(f"Common wrong category: {example['common_wrong_category']}")
    print("***")
    response = client.chat.completions.create(
        model="microsoft/Phi-3.5-mini-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["text"]},
        ],
        temperature=0,
        max_tokens=64,
    )

    print(f"MODEL RESPONSE: {response.choices[0].message.content}")

    print("-" * 30)

# %%
