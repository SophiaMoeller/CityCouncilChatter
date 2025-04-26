import numpy as np
from bertopic import BERTopic
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from gensim.corpora import Dictionary
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from preprocessing import preprocess, df, documents
from analysis import compute_coherence_values_lsa, compute_coherence_values_lda
from advanced_analysis import get_doc_vector, compute_kmeans_silhouette

# Preprocess all documents
preprocessed_docs = [preprocess(doc) for doc in documents]

texts = [" ".join(doc) for doc in preprocessed_docs]

# Create dictionary and corpus for Gensim coherence score
dictionary = Dictionary(preprocessed_docs)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

# Use bigrams for vectorization
bow_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bow = bow_vectorizer.fit_transform(texts)

tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, 2))
X_tfidf = tfidf_vectorizer.fit_transform(texts)

# Create bigram model for Word2Vec and BERTopic
bigram_model = Phrases(preprocessed_docs, min_count=5, threshold=10)
bigram_phraser = Phraser(bigram_model)

bigrammed_docs = [bigram_phraser[doc] for doc in preprocessed_docs]

# Join back for vectorizers and BERTopic
texts_with_bigrams = [" ".join(doc) for doc in bigrammed_docs]

# Word2Vec model
w2v_model = Word2Vec(sentences=bigrammed_docs, vector_size=100, window=5, min_count=2, workers=4)
X_w2v = np.array([get_doc_vector(doc, w2v_model) for doc in bigrammed_docs])

if __name__ == "__main__":

    start, limit, step = 2, 10, 1

    # Coherence Scores LDA
    print("\n LDA Coherence Scores:")
    lda_scores = compute_coherence_values_lda(dictionary, bigrammed_docs, start, limit, step)
    best_lda = max(lda_scores, key=lambda x: x[1])
    for num_topics, score in lda_scores:
        print(f"{num_topics} topics: Coherence Score = {score:.4f}")
    print(f" Best LDA topic count: {best_lda[0]} with score {best_lda[1]:.4f}")

    # Coherence Scores LSA
    print("\n LSA Coherence Scores:")
    lsa_scores = compute_coherence_values_lsa(dictionary, bigrammed_docs, start, limit, step)
    best_lsa = max(lsa_scores, key=lambda x: x[1])
    for num_topics, score in lsa_scores:
        print(f"{num_topics} topics: Coherence Score = {score:.4f}")
    print(f" Best LSA topic count: {best_lsa[0]} with score {best_lsa[1]:.4f}")

    # LSA Topics
    print("\n--- LSA Topics ---")
    lsa_model = TruncatedSVD(n_components=2, random_state=42)
    lsa_model.fit(X_tfidf)
    terms = tfidf_vectorizer.get_feature_names_out()
    for i, topic in enumerate(lsa_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
        print(f"Topic {i + 1}: {', '.join(top_terms)}")

    # LDA Topics
    print("\n--- LDA Topics ---")
    lda_model = LatentDirichletAllocation(n_components=6, max_iter=10, random_state=42)
    lda_model.fit(X_bow)
    terms = bow_vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
        print(f"Topic {idx + 1}: {', '.join(top_terms)}")

    # KMeans Clustering
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
        topic_docs = [bigrammed_docs[i] for i in range(len(bigrammed_docs)) if df['kmeans_topic'][i] == topic_num]
        all_words = [word for doc in topic_docs for word in doc]
        top_words = [word for word, _ in Counter(all_words).most_common(10)]
        doc_count = len(topic_docs)
        print(f"\n Topic {topic_num + 1} ({doc_count} documents):")
        print("Top words:", ", ".join(top_words))

    # BERTopic with HDBSCAN
    print("\n BERTopic Results:")
    texts1 = texts_with_bigrams
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
    topic_model = BERTopic(language="multilingual", embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(texts1)

    df['bertopic_topic'] = topics
    for topic_num in sorted(set(topics)):
        if topic_num == -1:
            continue
        top_words = topic_model.get_topic(topic_num)
        words = ", ".join([w for w, _ in top_words[:10]])
        count = topics.count(topic_num)
        print(f"\n BERTopic {topic_num} ({count} documents): {words}")
