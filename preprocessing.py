import os
import fitz
import re
import nltk
import spacy
import pandas as pd
from nltk.corpus import stopwords
import string

# Download NLTK resources
#nltk.download('punkt')
#nltk.download('stopwords')

# Set of stopwords and punctuation
stop_words = set(stopwords.words('german'))
punctuation = set(string.punctuation)

# Load German spaCy model
nlp = spacy.load("de_core_news_sm")

# Folder containing the downloaded PDFs
PDF_DIR = 'C:/Users/sophi/OneDrive - IU International University of Applied Sciences/IU/Project Data Analysis/Protokolle Stadtverordnetenversammlung'


# Custom domain-specific and person names stopwords
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


# Preprocessing and lemmatization function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zäöüß\s]', ' ', text)  # Keep only letters and whitespace

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
