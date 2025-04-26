from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
from gensim.models import Word2Vec
from preprocessing import df, documents, preprocess

# Preprocess documents
preprocessed_docs = [preprocess(doc) for doc in documents]

# Word2Vec Model
w2v_model = Word2Vec(sentences=preprocessed_docs, vector_size=100, window=5, min_count=2, workers=4)


def get_doc_vector(doc, model):
    """
    Computes the average Word2Vec embedding vector for a given document.
    Each document is represented as the mean of the vectors for all words in the document
    that exist in the Word2Vec model's vocabulary.

    :param doc: A tokenized document (list of words).
    :param model: A trained Word2Vec model.
    :return: A 1D vector (of length model.vector_size) representing the average embedding of the document. If no valid
    words are found, a zero vector is returned.
    """
    valid_words = [word for word in doc if word in model.wv]
    return np.mean([model.wv[word] for word in valid_words], axis=0) if valid_words else np.zeros(model.vector_size)


X_w2v = np.array([get_doc_vector(doc, w2v_model) for doc in preprocessed_docs])


def compute_kmeans_silhouette(X, start=2, limit=10):
    """
    Computes the silhouette score for KMeans clustering using different numbers of clusters.

    The silhouette score measures how similar each data point is to its own cluster compared to other clusters. It
    ranges from -1 (bad) to 1 (good), with higher scores indicating more distinct and well-formed clusters.
    :param X: The vectorized document representations (e.g., Word2Vec).
    :param start: The starting number of clusters to evaluate.
    :param limit: The maximum number of clusters to evaluate.
    :return: A list of tuples where each tuple is (num_clusters, silhouette_score).
    """
    results = []
    for k in range(start, limit):
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        results.append((k, score))
    return results


if __name__ == "__main__":
    # Silhouette Scores
    print("\n KMeans Word2Vec Silhouette Scores:")
    kmeans_scores = compute_kmeans_silhouette(X_w2v)
    for k, score in kmeans_scores:
        print(f"{k} clusters: Silhouette Score = {score:.4f}")

    # Choose the best k
    best_k = max(kmeans_scores, key=lambda x: x[1])[0]
    print(f"\n Best number of clusters based on silhouette score: {best_k}")

    # Fit KMeans with best_k
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    df['kmeans_topic'] = kmeans.fit_predict(X_w2v)

    # Display topic keywords
    for topic_num in range(best_k):
        topic_docs = [preprocessed_docs[i] for i in range(len(preprocessed_docs)) if df['kmeans_topic'][i] == topic_num]
        all_words = [word for doc in topic_docs for word in doc]
        top_words = [word for word, _ in Counter(all_words).most_common(10)]
        doc_count = len(topic_docs)
        print(f"\n Topic {topic_num + 1} ({doc_count} documents):")
        print("Top words:", ", ".join(top_words))

    # BERTopic with HDBSCAN
    print("\n BERTopic Results:")
    texts = [" ".join(doc) for doc in preprocessed_docs]
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    topic_model = BERTopic(language="multilingual", embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(texts)

    df['bertopic_topic'] = topics
    for topic_num in sorted(set(topics)):
        if topic_num == -1:
            continue
        top_words = topic_model.get_topic(topic_num)
        words = ", ".join([w for w, _ in top_words[:10]])
        count = topics.count(topic_num)
        print(f"\n BERTopic {topic_num} ({count} documents): {words}")
