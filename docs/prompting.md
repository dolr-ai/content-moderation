## 1. Instruction-Based + Few-Shot Prompting
Combine explicit rules with examples of both acceptable and violating content to train the model to recognize nuances.
**Example**:
```
Prompt:
"Check if this text violates our content policy (no hate speech, threats, or harassment).
Examples of violations:
- 'I hate [group] and they should all leave.' → VIOLATION
- 'You’re an idiot.' → VIOLATION
Examples of non-violations:
- 'I disagree with your opinion.' → OK
- 'This is frustrating.' → OK

Text to check: 'People like you deserve to suffer.'"
Output: "VIOLATION (threat/hostility)."
```
**Why it works**:
- Provides clear guidelines (**instruction-based**).
- Teaches context with labeled examples (**few-shot**).

----

## 2. Chain-of-Thought (CoT) Prompting
For complex cases, ask the model to explain its reasoning to reduce false positives/negatives.
**Example**:
```
Prompt:
"Analyze whether this comment violates our policy against harassment. Think step by step:
Comment: 'Your ideas are trash, and you should quit your job.'
Step 1: Does it attack a person/group? → Yes ('your ideas are trash').
Step 2: Is it a threat? → No.
Step 3: Does it qualify as harassment? → Subjective, but likely yes (personal attack).
Final verdict: VIOLATION."
```
**Why it works**:
- Forces the model to break down moderation logic.
- Useful for appeals or transparency.

----

## 3. Hybrid Prompting
Combine structured rules with examples to improve accuracy.
**Example**:
```

