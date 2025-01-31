# Category Map

```
PRIMARY_CATEGORY_MAP = {
    "clean": 0,
    "hate_or_discrimination": 1,
    "violence_or_threats": 2,
    "offensive_language": 3,
    "nsfw_content": 4,
    "spam_or_scams": 5,
}
```

```
category_mapping = {
    # Hate or discrimination
    "hate": "hate_or_discrimination",
    "hate/threatening": "hate_or_discrimination",
    "harassment": "hate_or_discrimination",
    "harassment/threatening": "hate_or_discrimination",

    # Violence or threats
    "violence": "violence_or_threats",
    "violence/graphic": "violence_or_threats",
    "illicit/violent": "violence_or_threats",
    "self-harm": "violence_or_threats",
    "self-harm/intent": "violence_or_threats",
    "self-harm/instructions": "violence_or_threats",

    # NSFW content
    "sexual": "nsfw_content",
    "sexual/minors": "nsfw_content",

    # ignore spam_or_scams & offensive_language for now
    "illicit": "spam_or_scams",
}

```

# Metrics

Overall Metrics:
accuracy: 0.5086
macro_precision: 0.3892
macro_recall: 0.4554
macro_f1: 0.3887

Per-Category Metrics:

clean:
precision: 0.4642
recall: 0.9890
f1: 0.6319
support: 3000.0000
accuracy: 0.9890

hate_or_discrimination:
precision: 0.5629
recall: 0.9045
f1: 0.6939
support: 2000.0000
accuracy: 0.9045

violence_or_threats:
precision: 0.6776
recall: 0.5421
f1: 0.6023
support: 380.0000
accuracy: 0.5421

offensive_language:
precision: 0.0000
recall: 0.0000
f1: 0.0000
support: 2000.0000
accuracy: 0.0000

nsfw_content:
precision: 0.6306
recall: 0.2970
f1: 0.4038
support: 1000.0000
accuracy: 0.2970

spam_or_scams:
precision: 0.0000
recall: 0.0000
f1: 0.0000
support: 2000.0000
accuracy: 0.0000

# hate_or_discrimination misclassifications

## Analysis of False Positives/Negatives of Hate/Discrimination

**False Negative** = Content that contains hate or discrimination but was incorrectly classified as clean
**False Positive** = Content that is clean but was incorrectly classified as containing hate or discrimination

## Summary

The model struggles with identifying hateful content when it's expressed through coded language, slang variations, or cultural references, while sometimes misclassifying strong disagreements or criticisms as hate speech.

## Analysis

### False Negatives (Missed Hate/Discrimination)

### 1. Modified Slurs and Coded Language

The model fails to identify **modified versions** of slurs and **coded discriminatory language**:

- **Modified spellings**:
  - `"nicca/niccas"`
  - `"niger"`
  - `"coons"`
- **Coded references**:
  - `"colored folk"`
  - `"cracker"`
  - `"ghetto school"`

### 2. Cultural/Ethnic Discrimination

The model misses **discriminatory statements** targeting specific groups:

- **Racial stereotypes**:
  - `"Dis broad done tagged dis nicca n a video"`
  - `"ching chong"`

### 3. Identity-Based Attacks

The model fails to catch **derogatory language** targeting identity characteristics:

- **Gender-based discrimination**:
  - `"hoes pick me like dandelions"`
  - `"smh queer"`
- **Racial prejudice**:
  - `"blame whitey for EVERYTHING!"`
  - `"border jumper"`

### False Positives (Incorrectly Flagged Clean Content)

### 1. Strong Disagreements

The model incorrectly flags **heated discussions** and **strong criticism**:

- **Editorial disputes**:
  - `"Your accusations of bad faith editing are as full of bullshit now as when you made them"`
  - `"you're taking things too far"`
- **Cultural debates**:
  - `"National agenda!!! Hahahahaha! Who are the two greeks writing gossip"`
  - `"Could you please look at the article of Iranian people?"`

### 2. Personal Conflicts

The model misinterprets **interpersonal conflict** as hate speech:

- **Direct confrontations**:
  - `"Let's have a talk, Abby!"`
  - `"Mate, I have no idea who you are. Go away"`

## Potential Causes of Misclassification

### 1. Context Understanding Limitations

The model shows difficulties with:

- Distinguishing between heated discussion and genuine hate speech
- Understanding cultural and historical context that makes certain terms offensive
- Identifying subtle forms of discrimination and coded language

### 2. Language Pattern Recognition

The model struggles with:

- Recognizing variations and modifications of discriminatory terms
- Understanding context-dependent offensiveness
- Differentiating between strong criticism and discriminatory attacks

# nsfw_content misclassifications

## Analysis of False Positives/Negatives of NSFW Content

**False Negative** = Content that contains NSFW material but was incorrectly classified as clean
**False Positive** = Content that is clean but was incorrectly classified as containing NSFW material

## Summary

The model struggles to identify suggestive content that uses indirect or metaphorical language, while sometimes misclassifying innocent content containing certain keywords or anatomical references.

## Analysis

### False Negatives (Missed NSFW Content)

### 1. Suggestive Language

The model fails to identify **implicit adult content** through suggestive phrasing:

- **Euphemistic expressions**:
  - `"I love losing control with you inside me"`
  - `"I want to see you completely undone"`
- **Implicit suggestions**:
  - `"What would you do if I was lying next to you?"`
  - `"How do you like to be touched?"`

### 2. Romantic/Physical Description

The model misses **suggestive descriptions** of physical interaction:

- **Physical intimacy**:
  - `"I love feeling your heartbeat through your chest"`
  - `"I want to feel your body against mine"`
- **Romantic advances**:
  - `"Wish I could kiss you before bed"`
  - `"I want to feel your lips whispering secrets"`

### 3. Adult Content Solicitation

The model fails to identify **adult content marketing**:

- **Service offerings**:
  - `"I understand. We all go through this at times. Just understand that this is my job"`
  - `"Do you want to know what's on my mind?"`

## Potential Causes of Misclassification

### 1. Context Understanding Limitations

The model shows difficulties with:

- Distinguishing between romantic and explicit content
- Understanding euphemistic or metaphorical language
- Identifying context that makes certain phrases suggestive

### 2. Intent Recognition

The model struggles with:

- Differentiating between legitimate and suggestive physical descriptions
- Understanding the context that makes certain phrases inappropriate
- Identifying adult content marketing when phrased indirectly

# violence_or_threats misclassifications

## Analysis of False Positives/Negatives of Violence/Threats

**False Negative** = Content that contains violence or threats but was incorrectly classified as clean
**False Positive** = Content that is clean but was incorrectly classified as containing violence or threats

## Summary

The model struggles to identify indirect threats and veiled violent content, while sometimes misinterpreting figurative language or strong disagreements as threats.

## Analysis

### False Negatives (Missed Violent/Threatening Content)

### 1. Indirect Threats

The model fails to identify **veiled threats** and **implied violence**:

- **Passive aggressive threats**:
  - `"Unless you wish to be targeted yourself in a similar manner to Deskana I suggest to cease stalking me"`
  - `"Don't ever show your face where I live and don't ever let our paths cross"`
- **Conditional threats**:
  - `"If you block me for telling you what you are, there will be a holy war"`
  - `"if you keep going with this Nationalist propaganda Chauvinism and Hate speech, we might start looking for you"`

### 2. Death Threats/Wishes

The model misses **explicit death-related threats**:

- **Direct statements**:
  - `"and said he was going to kill me"`
  - `"death to vandalist of this Maratha sport"`
- **Death wishes**:
  - `"Drop dead"`
  - `"hey!!!!!!! thank you for re-editing my contributions it was really kind of you!!!!!!! i hope you get bowel cancer"`

### False Positives (Incorrectly Flagged Clean Content)

### 1. Figurative Language

The model incorrectly flags **metaphorical expressions** and **common phrases**:

- **Casual expressions**:
  - `"I am evil, hear me roar!"`
  - `"If no-one beats me to it, I'll knock something up tonight"`

### 2. Gaming/Entertainment Context

The model misinterprets **gaming or entertainment references** as threats:

- **Game discussions**:
  - References to combat in video games
  - Discussions of military history or weapons in an academic context

## Potential Causes of Misclassification

### 1. Context Understanding Limitations

The model shows difficulties with:

- Distinguishing between actual threats and figurative language
- Understanding when violent terms are used in legitimate contexts
- Identifying passive-aggressive or implied threats

### 2. Intent Recognition

The model struggles with:

- Differentiating between casual expressions and genuine threats
- Understanding context that makes certain phrases threatening
- Identifying threats that use indirect or coded language

---

# ~~offensive_language misclassifications~~

Skipped because this category is not supported in openai’s moderation

# ~~spam_or_scams misclassifications~~

Skipped because this category is not supported in openai’s moderation
