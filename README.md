# content-moderation

Content Moderation Pipeline

# datasets for benchmarking

## Jigsaw Toxic Comment

url: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/

| id               | comment_text                                                                                   | toxic | severe_toxic | obscene | threat | insult | identity_hate |
| ---------------- | ---------------------------------------------------------------------------------------------- | ----- | ------------ | ------- | ------ | ------ | ------------- |
| 0000997932d777bf | Explanation Why the edits made under my username Hardcore Metallica Fan were reverted? They... | 0     | 0            | 0       | 0      | 0      | 0             |
| 000103f0d9cfb60f | D'aww! He matches this background colour I'm seemingly stuck with. Thanks...                   | 0     | 0            | 0       | 0      | 0      | 0             |

## Sentiment140

url: https://www.kaggle.com/datasets/kazanova/sentiment140/data

- target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- ids: The id of the tweet ( 2087)
- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- flag: The query (lyx). If there is no query, then this value is NO_QUERY.
- user: the user that tweeted (robotickilldozr)
- text: the text of the tweet (Lyx is cool)

| target | ids  | date                         | flag     | user            | text                                                                                                                       |
| ------ | ---- | ---------------------------- | -------- | --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 0      | 2087 | Sat May 16 23:58:44 UTC 2009 | NO_QUERY | _TheSpecialOne_ | @switchfoot [link](http://twitpic.com/2y1zl) - Awww, that's a bummer. You shoulda got David Carr of Third Day to do it. ;D |
| 0      | 2087 | Sat May 16 23:58:44 UTC 2009 | NO_QUERY | scotthamilton   | is upset that he can't update his Facebook by texting it... and might cry as a result. School today also. Blah!            |
| 0      | 2087 | Sat May 16 23:58:44 UTC 2009 | NO_QUERY | mattycus        | @Kenichan I dived many times for the ball. Managed to save 50%. The rest go out of bounds.                                 |

## Hate Speech and Offensive Language

url: https://github.com/t-davidson/hate-speech-and-offensive-language/blob/master/data/labeled_data.csv

- count: Number of CrowdFlower users who coded each tweet (minimum is 3; sometimes more users coded a tweet when judgments were determined to be unreliable by CF).
- hate_speech: Number of CF users who judged the tweet to be hate speech.
- offensive_language: Number of CF users who judged the tweet to be offensive.
- neither: Number of CF users who judged the tweet to be neither offensive nor non-offensive.
- class: Class label for the majority of CF users (0 = Hate speech, 1 = Offensive language, 2 = Neither)

| count | hate_speech | offensive_language | neither | class | tweet                                                                                                                                        |
| ----- | ----------- | ------------------ | ------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | 3           | 0                  | 3       | 2     | !!! RT @mayasolovely: As a woman you shouldn't complain about cleaning up your house. &amp; as a man you should always take the trash out... |
| 1     | 3           | 0                  | 3       | 1     | !!!!! RT @mleew17: boy dats cold...tyga dwn bad for cuffin dat hoe in the 1st place!!                                                        |

## Reddit Comments May 2015

url : https://www.kaggle.com/datasets/kaggle/reddit-comments-may-2015

The database has one table, May2015, with the following fields:

created*utc: The timestamp (in UTC) when the comment was created.
ups: The number of upvotes the comment received.
subreddit_id: The unique identifier for the subreddit where the comment was posted.
link_id: The unique identifier for the post to which the comment belongs.
name: The unique name of the comment, typically in the format "t1*<comment_id>".
score_hidden: A boolean indicating whether the score of the comment is hidden.
author_flair_css_class: The CSS class for the author's flair, which can be used for styling.
author_flair_text: The text displayed in the author's flair.
subreddit: The name of the subreddit where the comment was posted.
id: The unique identifier for the comment.
removal_reason: The reason for the comment's removal, if applicable.
gilded: The number of times the comment has been gilded (given gold awards).
downs: The number of downvotes the comment received.
archived: A boolean indicating whether the comment is archived.
author: The username of the comment's author.
score: The net score of the comment (upvotes minus downvotes).
retrieved_on: The timestamp when the comment was retrieved from the database.
body: The content of the comment.
distinguished: A boolean indicating whether the comment is distinguished (e.g., by moderators).
edited: A boolean indicating whether the comment has been edited.
controversiality: A score indicating how controversial the comment is.
parent_id: The unique identifier of the parent comment or post, if applicable.

## Sarcastic Comments on Reddit

url: https://www.kaggle.com/datasets/sherinclaudia/sarcastic-comments-on-reddit

| label | comment                                                                    | author    | subreddit | score | ups | downs | date    | created_utc         | parent_comment                                                                                                                         |
| ----- | -------------------------------------------------------------------------- | --------- | --------- | ----- | --- | ----- | ------- | ------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | NC and NH.                                                                 | Trumpbart | politics  | 2     | -1  | -1    | 2016-10 | 2016-10-16 23:55:23 | Yeah, I get that argument. At this point, I'd prefer is she lived in NC as well.                                                       |
| 0     | You do know west teams play against west teams more than east teams right? | Shbshb906 | nba       | -4    | -1  | -1    | 2016-11 | 2016-11-01 00:24:10 | The blazers and Mavericks (The wests 5 and 6 seed) did not even carry a good enough record to make the playoffs in the east last year. |
