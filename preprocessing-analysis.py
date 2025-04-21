import os
import fitz
import re
import nltk
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import string
import numpy as np

# Download NLTK resources (only once)
#nltk.download('punkt')
#nltk.download('stopwords')

# Set of stopwords and punctuation
stop_words = set(stopwords.words('german'))
punctuation = set(string.punctuation)

# Custom domain-specific stopwords
def load_custom_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    words_extended = set()
    for word in words:
        words_extended.add(word.lower())
        words_extended.add(word.capitalize())
    return words_extended

def load_name_list(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

custom_stopwords = load_custom_stopwords("custom_stopwords.txt")
static_name_list = load_custom_stopwords("german_names.txt")

# Load German spaCy model
nlp = spacy.load("de_core_news_sm")

# Folder containing your downloaded PDFs
PDF_DIR = 'C:/Users/sophi/OneDrive - IU International University of Applied Sciences/IU/Project Data Analysis/Protokolle Stadtverordnetenversammlung'


# Preprocessing and lemmatization function
def preprocess(text):
    # Lowercase and basic cleanup
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', ' ', text)  # Keep only letters and whitespace

    # Use spaCy to tokenize and lemmatize
    doc = nlp(text)
    person_names = set(ent.text.lower() for ent in doc.ents if ent.label_ in ("PER", "PERSON"))
    tokens = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha  # Only alphabetic tokens
           and not token.is_stop  # Remove stopwords
           and token.lemma_.lower() not in custom_stopwords  # Remove custom stopwords
           and token.lemma_.lower() not in person_names
           and token.lemma_.lower() not in static_name_list
           and len(token) > 2  # Remove short tokens
    ]
    return tokens


# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()
    return full_text.strip()


# Collect all PDF texts
documents = []
file_names = []

for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        full_path = os.path.join(PDF_DIR, file)
        try:
            text = extract_text_from_pdf(full_path)
            documents.append(text)
            file_names.append(file)
        except Exception as e:
            print(f"Failed to process {file}: {e}")

# Create DataFrame
df = pd.DataFrame({'filename': file_names, 'document': documents})
print(f"\nLoaded {len(df)} documents.\n")
print(df)

# Vectorization with custom tokenizer
bow_vectorizer = CountVectorizer(tokenizer=preprocess)
X_bow = bow_vectorizer.fit_transform(df['document'])

tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess)
X_tfidf = tfidf_vectorizer.fit_transform(df['document'])

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

# --- LSA (using TF-IDF) ---
print("\n--- LSA Topics ---")
lsa_model = TruncatedSVD(n_components=5, random_state=42)
lsa_model.fit(X_tfidf)
terms = tfidf_vectorizer.get_feature_names_out()

for i, topic in enumerate(lsa_model.components_):
    top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
    print(f"Topic {i+1}: {', '.join(top_terms)}")

# --- LDA (using Count Vectorizer) ---
print("\n--- LDA Topics ---")
lda_model = LatentDirichletAllocation(n_components=5, max_iter=10, random_state=42)
lda_model.fit(X_bow)
terms = bow_vectorizer.get_feature_names_out()

for idx, topic in enumerate(lda_model.components_):
    top_terms = [terms[i] for i in topic.argsort()[-10:][::-1]]
    print(f"Topic {idx+1}: {', '.join(top_terms)}")
