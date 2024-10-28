import nltk
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter, defaultdict

# Load the first 1000 sentences from the Brown corpus
corpus_sentences = [' '.join(sent) for sent in brown.sents()[:1000]]
corpus_tokens = brown.sents()[:1000]


# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the corpus sentences
tfidf_matrix = vectorizer.fit_transform(corpus_sentences)

# Get feature names
feature_names = vectorizer.get_feature_names_out()

# Function to get the TF-IDF score of a specific word in a document
def get_tfidf_for_word(word, document_index):
    try:
        word_index = feature_names.tolist().index(word)
        return tfidf_matrix[document_index, word_index]
    except ValueError:
        return 0  # If the word is not in the vocabulary

# Get the TF-IDF for the first document (corpus[0])
doc_index = 0
words_to_check = ["county", "investigation", "produced"]

# Display TF-IDF values for the specified words
for word in words_to_check:
    tfidf_value = get_tfidf_for_word(word, doc_index)
    print(f"TF-IDF for word '{word}' in the first document: \n{tfidf_value:.4f}")

# Define the window size for context
window_size = 5

# Compute co-occurences
def get_cooccurrence_counts(corpus, window_size):
    word_count = Counter()
    cooccurrence_count = defaultdict(Counter)
    
    for sentence in corpus:
        for i, word in enumerate(sentence):
            word_count[word] += 1
            context = sentence[max(0, i - window_size): i] + sentence[i + 1: i + 1 + window_size]
            for context_word in context:
                cooccurrence_count[word][context_word] += 1
    return word_count, cooccurrence_count

# Compute PPMI
def compute_ppmi(word_count, cooccurrence_count, total_words):
    ppmi = defaultdict(Counter)
    for word, context_words in cooccurrence_count.items():
        for context_word, co_count in context_words.items():
            prob_word = word_count[word] / total_words
            prob_context_word = word_count[context_word] / total_words
            prob_cooccurrence = co_count / total_words
            if prob_cooccurrence > 0:
                pmi = np.log2(prob_cooccurrence / (prob_word * prob_context_word))
                ppmi[word][context_word] = max(pmi, 0)  # PPMI is max(0, PMI)
    return ppmi

# Calculate word and co-occurrence counts
word_count, cooccurrence_count = get_cooccurrence_counts(corpus_tokens, window_size)

# Total number of words in the corpus
total_words = sum(word_count.values())

# Compute PPMI values
ppmi_matrix = compute_ppmi(word_count, cooccurrence_count, total_words)

# Display PPMI for specific word-context pairs
word_context_pairs = [("expected", "approve"), ("mentally", "in"), ("send", "bed")]

print("\nPPMI for a few word pairs:")
for word, context_word in word_context_pairs:
    ppmi_value = ppmi_matrix[word][context_word]
    print(f"PPMI ('{word}', '{context_word}') = \n{ppmi_value:.4f}")
