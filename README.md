# 🏛️ NLP Topic Modeling on City Council Meeting Protocols

This project explores the application of Natural Language Processing (NLP) techniques—particularly **topic modeling**—to real-world, unstructured text data from the **Stadtverordnetenversammlung (city council meeting) of Kassel** protocols. The goal is to extract meaningful topics, patterns, and themes from official documents, contributing to transparency and digital civic analysis.

---

## 📄 Project Description

City council meeting minutes are rich in information but often inaccessible for automated analysis due to their unstructured nature. This project focuses on:

- Extracting text from publicly available PDF transcripts.
- Preprocessing the raw German text with proper linguistic cleaning.
- Transforming the cleaned data into numerical vector formats.
- Applying **Latent Dirichlet Allocation (LDA)** and **Latent Semantic Analysis (LSA)** to uncover the most prevalent topics.

---
## 📚 Data Source

The primary data source for this project are publicly available **PDF protocols** from the **Stadtverordnetenversammlung** (city council meeting) held in **Kassel, Germany** between 2015 and 2025. The documents include formal discussions, proposals, and citizen-relevant topics, making it a valuable example of real-world unstructured text in the public administration domain.

- 📄 **Title**: *Öffentliche Niederschrift Stadtverordnetenversammlung*  
- 🗓️ **Date**: 2015-2025 
- 🗃️ **Format**: PDF  
- 🌐 **Source**: [City of Kassel Website](https://ratsinfo.kassel.de/sdnet4/termine/?__=UGhVM0hpd2NXNFdFcExjZcYv9COU5nx2dvbiny8_tjGcE7vQyaPX4JQEhAS-I4V_)

In accordance with ethical web scraping procedures the `robots.txt` file was checked before downloading the individual files.
___

## 🧰 Technologies Used

- **Python 3.10+**
- **PyMuPDF (`fitz`)** – for PDF text extraction
- **spaCy (`de_core_news_sm`)** – for German lemmatization
- **NLTK** – for stopword handling
- **scikit-learn** – for vectorization and topic modeling
- **pandas, numpy** – for data manipulation and matrix operations

---

## 🧼 Preprocessing Steps

Text cleaning includes:
- Lowercasing
- Removing non-alphabetic characters
- Removing German stopwords
- Lemmatizing with spaCy
- Filtering short tokens

---

## 🔢 Vectorization Techniques

- **Bag of Words (BoW)**
- **TF-IDF (Term Frequency-Inverse Document Frequency)**  
- N-gram ranges (e.g., unigrams to trigrams) can be configured for deeper semantic patterns.

---

## 🔍 Topic Modeling

### ✅ Latent Dirichlet Allocation (LDA)
- Based on probabilistic topic distributions
- Applied to BoW vectors

### ✅ Latent Semantic Analysis (LSA)
- Based on truncated SVD
- Applied to TF-IDF vectors

Both methods extract top keywords per topic to interpret underlying themes.

---

## 🚀 How to Run

Install dependencies:
`pip install -r requirements.txt`
`python -m spacy download de_core_news_sm`

As the protocols are in German language it is necessary to install the respective language program for the preprocessing of the data.