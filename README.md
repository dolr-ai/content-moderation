# Content Moderation

Content moderation pipeline benchmarking datasets.

## Datasets

### Jigsaw Toxic Comment Classification

Source: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/

Multi-label classification of toxic online comments. Labels include: toxic, severe_toxic, obscene, threat, insult, and identity_hate.

```json
{
  "id": "0000997932d777bf",
  "comment_text": "Explanation Why the edits made under my username...",
  "toxic": 0,
  "severe_toxic": 0,
  "obscene": 0,
  "threat": 0,
  "insult": 0,
  "identity_hate": 0
}
```

### Sentiment140

Source: https://www.kaggle.com/datasets/kazanova/sentiment140/data

Twitter sentiment analysis dataset with 1.6 million tweets labeled for sentiment (0 = negative, 2 = neutral, 4 = positive).

```json
{
  "target": 0,
  "ids": 2087,
  "date": "Sat May 16 23:58:44 UTC 2009",
  "flag": "NO_QUERY",
  "user": "_TheSpecialOne_",
  "text": "@switchfoot - Awww, that's a bummer..."
}
```

### Hate Speech and Offensive Language

Source: https://github.com/t-davidson/hate-speech-and-offensive-language/

Crowdsourced dataset of tweets labeled for hate speech, offensive language, or neither.

```json
{
  "count": 0,
  "hate_speech": 3,
  "offensive_language": 0,
  "neither": 3,
  "class": 2,
  "tweet": "!!! RT @mayasolovely: As a woman you shouldn't complain..."
}
```

Class labels:

- 0: Hate speech
- 1: Offensive language
- 2: Neither

### Financial News Sentiment

Source: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

Sample data format:

```json
{
  "text": "$BYND - JPMorgan reels in expectations on Beyond...",
  "label": 0
}
```

Class labels:

- 0: Negative
- 1: Neutral
- 2: Positive

### Scam Data

Source: https://huggingface.co/datasets/OtabekRizayev/scam-data

```json
{
  "text": "Important notice: Your prize claim requires urgent attention. Act without delay to avoid issues. Ref: 7701",
  "label": 1
}
```

Class labels:

- 0: Not a scam
- 1: Scam

### All Scam Spam

Source: https://huggingface.co/datasets/FredZhang7/all-scam-spam

```json
{
  "text": "Dear Voucher Holder, To claim this weeks offer, at your PC please go to http://www.wtlp.co.uk/text.
	daily news - rolex is real cheap out here ! - diehard owing8",
  "is_spam": 1
}
```
