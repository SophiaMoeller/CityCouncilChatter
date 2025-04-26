from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocessing import preprocess, df, documents

# Preprocess all documents
preprocessed_docs = [preprocess(doc) for doc in documents]

# Create dictionary and corpus from tokenized documents
dictionary = Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Vectorization with custom tokenizer
bow_vectorizer = CountVectorizer(tokenizer=preprocess)
tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess)

preprocessed_texts = [" ".join(doc) for doc in preprocessed_docs]

X_bow = bow_vectorizer.fit_transform(preprocessed_texts)
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_texts)

# Print a preview of the cleaned vocabulary
print("BOW features:", bow_vectorizer.get_feature_names_out()[:10])
print("TF-IDF features:", tfidf_vectorizer.get_feature_names_out()[:10])

# Display top 10 common BOW words
word_counts = X_bow.toarray().sum(axis=0)
feature_names = bow_vectorizer.get_feature_names_out()
top_10_bow = sorted(zip(feature_names, word_counts), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 most common words (BOW):")
for word, freq in top_10_bow:
    print(f"{word}: {freq}")

# Display top 10 TF-IDF words
tfidf_scores = X_tfidf.toarray().sum(axis=0)
feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
top_10_tfidf = sorted(zip(feature_names_tfidf, tfidf_scores), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 most important words (TF-IDF):")
for word, score in top_10_tfidf:
    print(f"{word}: {score:.4f}")


def get_sklearn_topics(model, feature_names, n_top_words=10):
    """
    Extracts the top n words for each topic from a fitted sklearn topic model (LDA or LSA).

    :param model: Fitted sklearn topic model (e.g., TruncatedSVD or LatentDirichletAllocation).
    :param feature_names: Array of feature names (vocabulary) corresponding to the model.
    :param n_top_words: Number of top words to extract per topic.
    :return: A list of lists, where each sublist contains the top words for one topic.
    """
    return [[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]] for topic in model.components_]


def compute_coherence_values_lda(dictionary, texts, start, limit, step):
    """
    Computes the coherence score for LDA models with varying numbers of topics.

    :param dictionary: Gensim dictionary created from preprocessed texts.
    :param texts: List of preprocessed documents (as lists of tokens).
    :param start: Minimum number of topics.
    :param limit: Maximum number of topics.
    :param step: Step size for the number of topics.
    :return: A list of tuples (num_topics, coherence_score) for each number of topics.
    """
    results = []
    for num_topics in range(start, limit, step):
        model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        model.fit(X_bow)
        topic_words = get_sklearn_topics(model, bow_vectorizer.get_feature_names_out())
        cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        results.append((num_topics, coherence))
    return results


def compute_coherence_values_lsa(dictionary, texts, start, limit, step):
    """
    Computes the coherence score for LSA models (TruncatedSVD) with varying numbers of topics.

    :param dictionary: Gensim dictionary created from preprocessed texts.
    :param texts: List of preprocessed documents (as lists of tokens).
    :param start: Minimum number of topics.
    :param limit: Maximum number of topics.
    :param step: Step size for the number of topics.
    :return: A list of tuples (num_topics, coherence_score) for each number of topics.
    """
    results = []
    for num_topics in range(start, limit, step):
        model = TruncatedSVD(n_components=num_topics, random_state=42)
        model.fit(X_tfidf)
        topic_words = get_sklearn_topics(model, tfidf_vectorizer.get_feature_names_out())
        cm = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence = cm.get_coherence()
        results.append((num_topics, coherence))
    return results


if __name__ == "__main__":

    # Set range of topics to test
    start, limit, step = 2, 10, 1

    # Run LDA coherence search
    print("\n LDA Coherence Scores:")
    lda_scores = compute_coherence_values_lda(dictionary, preprocessed_docs, start, limit, step)
    best_lda = max(lda_scores, key=lambda x: x[1])
    for num_topics, score in lda_scores:
        print(f"{num_topics} topics: Coherence Score = {score:.4f}")
    print(f" Best LDA topic count: {best_lda[0]} with score {best_lda[1]:.4f}")

    # Run LSA coherence search
    print("\n LSA Coherence Scores:")
    lsa_scores = compute_coherence_values_lsa(dictionary, preprocessed_docs, start, limit, step)
    best_lsa = max(lsa_scores, key=lambda x: x[1])
    for num_topics, score in lsa_scores:
        print(f"{num_topics} topics: Coherence Score = {score:.4f}")
    print(f" Best LSA topic count: {best_lsa[0]} with score {best_lsa[1]:.4f}")

    # LSA (using TF-IDF)
    print("\n--- LSA Topics ---")
    lsa_model = TruncatedSVD(n_components=5, random_state=42)
    lsa_model.fit(X_tfidf)
    terms = tfidf_vectorizer.get_feature_names_out()

    for i, topic in enumerate(lsa_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
        print(f"Topic {i + 1}: {', '.join(top_terms)}")

    # LDA (using Count Vectorizer)
    print("\n--- LDA Topics ---")
    lda_model = LatentDirichletAllocation(n_components=3, max_iter=10, random_state=42)
    lda_model.fit(X_bow)
    terms = bow_vectorizer.get_feature_names_out()

    for idx, topic in enumerate(lda_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
        print(f"Topic {idx + 1}: {', '.join(top_terms)}")
