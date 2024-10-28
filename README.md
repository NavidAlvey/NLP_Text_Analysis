# Text Analysis with TF-IDF and PPMI

This project analyzes word significance and word relationships in a text corpus using TF-IDF and Positive Pointwise Mutual Information (PPMI). The Brown corpus from NLTK is used as the text source.
## Description
### Data Preparation:
- Loads the first 1000 sentences from the Brown corpus.
- Converts each sentence into a single string for TF-IDF analysis.
 
### TF-IDF Calculation:
- Initializes a TfidfVectorizer to compute TF-IDF scores for each word in each sentence.
- Defines a function to retrieve the TF-IDF score of a specific word in a specified document.

### Co-Occurrence and PPMI Computation:
- Calculates word co-occurrence counts within a defined window (5 words around each target word).
- Computes Positive Pointwise Mutual Information (PPMI) for word-context pairs based on co-occurrence probabilities.

## Requirements
- Python 3.x
- NLTK and scikit-learn libraries
- Download the Brown corpus in NLTK:
  ```python
  import nltk
  nltk.download('brown')
  ```
## Usage
### Calculate TF-IDF Scores for Specific Words
```python
tfidf_value = get_tfidf_for_word("example_word", document_index=0)
print(f"TF-IDF for 'example_word': {tfidf_value:.4f}")
```
### Compute PPMI Values for Word Pairs
```python
ppmi_value = ppmi_matrix["word"]["context_word"]
print(f"PPMI ('word', 'context_word') = {ppmi_value:.4f}")
```
