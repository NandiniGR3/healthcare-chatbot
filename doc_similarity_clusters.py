# ----------------------------------------
# NLP: Text Clustering & Document Similarity
# Datasets: AG News, BBC News
# TF-IDF + Cosine Similarity
# DBSCAN & GMM
# ----------------------------------------

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from datasets import load_dataset

# ----------------------------------------
# Helper function: load or save dataset
# ----------------------------------------
def load_or_save_dataset(dataset_name, split, size, file_name, text_key):
    if os.path.exists(file_name):
        print(f"Loading {file_name} from local storage...")
        df = pd.read_csv(file_name)
        return df[text_key].tolist()
    else:
        print(f"Downloading {dataset_name} from Hugging Face...")
        ds = load_dataset(dataset_name, split=split)
        ds = ds.select(range(size))
        documents = [item[text_key] for item in ds]
        pd.DataFrame({text_key: documents}).to_csv(file_name, index=False)
        print(f"{file_name} saved locally")
        return documents

# ----------------------------------------
# Load datasets
# ----------------------------------------

# AG News
documents_ag = load_or_save_dataset(
    dataset_name="ag_news",
    split="train",
    size=800,
    file_name="ag_news.csv",
    text_key="text"
)

# BBC News (FIXED DATASET NAME)
documents_bbc = load_or_save_dataset(
    dataset_name="SetFit/bbc-news",
    split="train",
    size=200,
    file_name="bbc_news.csv",
    text_key="text"
)

# Combine datasets
documents = documents_ag + documents_bbc
print("\nTotal Documents:", len(documents))

# ----------------------------------------
# TF-IDF Vectorization
# ----------------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.6,
    min_df=3
)

tfidf_matrix = vectorizer.fit_transform(documents)
print("TF-IDF Shape:", tfidf_matrix.shape)

# ----------------------------------------
# Cosine Similarity
# ----------------------------------------
cos_sim = cosine_similarity(tfidf_matrix)
print("\nCosine Similarity Sample (3x3):")
print(cos_sim[:3, :3])

# ----------------------------------------
# DBSCAN Clustering
# ----------------------------------------
dbscan = DBSCAN(
    eps=0.75,
    min_samples=5,
    metric='cosine'
)

dbscan_labels = dbscan.fit_predict(tfidf_matrix)
print("\nDBSCAN Cluster Distribution:")
print(pd.Series(dbscan_labels).value_counts())

# ----------------------------------------
# Gaussian Mixture Model (GMM)
# ----------------------------------------
tfidf_dense = tfidf_matrix.toarray()

gmm = GaussianMixture(
    n_components=6,
    random_state=42
)

gmm_labels = gmm.fit_predict(tfidf_dense)
print("\nGMM Cluster Distribution:")
print(pd.Series(gmm_labels).value_counts())

# ----------------------------------------
# Prediction on New Documents
# ----------------------------------------
new_docs = [
    "The government announced a new economic policy",
    "The football match ended in a thrilling victory",
    "Artificial intelligence is transforming data science"
]

new_tfidf = vectorizer.transform(new_docs)
predicted_clusters = gmm.predict(new_tfidf.toarray())

print("\nPrediction for New Documents (GMM):")
for doc, label in zip(new_docs, predicted_clusters):
    print(f"Cluster {label} --> {doc}")

# ----------------------------------------
# Results Table
# ----------------------------------------
results = pd.DataFrame({
    "Document": documents,
    "DBSCAN_Cluster": dbscan_labels,
    "GMM_Cluster": gmm_labels
})

print("\nSample Results:")
print(results.head())
