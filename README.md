# Rumor Stance Detection with Twitter-RoBERTa

Improving rumor stance detection on the **SemEval-2017 Task 8 (RumourEval)** dataset by replacing the Branch-LSTM state-of-the-art with a domain-specific transformer, contextual branch encoding, and class-balanced training.

## Overview

Stance detection is a NLP technique that identifies a person’s attitude/position regarding a specific topic. The SemEval-2017 Task 8 dataset contains a series of twitter threads regarding many different news events, which provides a benchmark for this task. Because of the way this dataset is constructed, a model needs to evaluate the stance of replies that can be nested in several tweet depths from the original source tweet. 

The SOTA for this dataset as linked on [NLP Progress](http://nlpprogress.com/english/stance_detection.html) utilizes an LSTM-based approach to model a conversation tree where the stance of an individual tweet depends on previous tweets in the tree. This Branch-LSTM approach allows the model to see the parent’s hidden state and even earlier context. However, it relies on static word embeddings and struggles on underrepresented classes, namely predicting **zero** "Deny" tweets in its reported confusion matrix.

This project targets those limitations directly.

## Approach

Our method combines three ideas:

1. **Domain-specific pre-training.** Instead of fine-tuning a general-purpose language model, we use [`cardiffnlp/twitter-roberta-base`](https://huggingface.co/cardiffnlp/twitter-roberta-base), a RoBERTa model pre-trained on ~58M tweets. This gives the encoder an inherent understanding of platform-specific phenomena such as hashtags, emojis, slang, and sarcasm — cues that frequently signal stance but are often missed by models trained on general web text.
2. **Contextual branch encoding.** For each target tweet, we concatenate its preceding branch (source tweet → ancestor replies → target) into a single input sequence. This preserves conversational order in the same spirit as the Branch-LSTM, but unlike bag-of-words / averaged static embeddings it also preserves **word order within tweets** — so "man eats dog" is no longer representationally identical to "dog eats man".
3. **Weighted Cross-Entropy Loss.** RumourEval-2017 is heavily imbalanced: the "Comment" class makes up roughly 65% of all labels, while Support / Deny / Query (SDQ) are comparatively rare. We apply class-weighted cross-entropy so the model pays disproportionate attention to these high-value SDQ signals during training.

## Dataset

- **SemEval-2017 Task 8, Subtask A** (RumourEval)
- Four stance labels per tweet: `Support`, `Deny`, `Query`, `Comment` (SDQC)
- Tweets are organized as conversation trees rooted at a source tweet; each reply inherits a path of ancestors that forms its "branch".

## Evaluation

- Primary metric: **accuracy** on the official RumourEval-2017 test split (for direct comparability with the SOTA).
- Secondary analysis: **per-class confusion matrix**, with particular attention to recall on `Deny`, `Support`, and `Query`.